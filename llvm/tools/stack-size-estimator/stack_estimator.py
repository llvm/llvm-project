#!/usr/bin/env python3

import argparse
import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Set

import call_graph
import graph_helper
import symbolizer
import extract_type_map
import consolidate_sorted_table

# Type aliases
SccResults = Dict[int, Dict[str, Any]]
FunctionInfoByAddr = Dict[int, Dict[str, Any]]


def calculate_max_stack_usage(
    cg: graph_helper.Graph,
    function_info_map: FunctionInfoByAddr,
    sccs: List[List[int]],
) -> SccResults:
    """
    Calculates max stack usage and tracks the path for each SCC.
    """
    node_to_scc_id = {node: i for i, scc in enumerate(sccs) for node in scc}

    # Build condensation graph
    scc_dag = graph_helper.Graph()
    for u in cg.vertices:
        scc_id_u = node_to_scc_id.get(u)
        if scc_id_u is None:
            continue
        scc_dag.add_vertex(scc_id_u)

        if u in cg.graph:
            for v in cg.graph[u]:
                scc_id_v = node_to_scc_id.get(v)
                if scc_id_v is not None and scc_id_u != scc_id_v:
                    scc_dag.add_edge(scc_id_u, scc_id_v)

    # Calculate local stack usage for each SCC
    scc_stack_usage: Dict[int, int] = {
        i: sum(
            function_info_map.get(func_addr, {}).get("stack_usage", 0)
            for func_addr in scc)
        for i, scc in enumerate(sccs)
    }

    max_stack_per_scc: Dict[int, int] = {}
    scc_next_hop: Dict[int, Optional[int]] = {}

    # Iterate through SCCs in reverse topological order
    sorted_sccs = scc_dag.topological_sort()

    for scc_id in reversed(sorted_sccs):
        successors = scc_dag.graph.get(scc_id, [])
        max_downstream_usage = 0
        best_next_scc = None

        if successors:
            # Find the successor with the maximum stack usage
            best_next_scc = max(
                successors, key=lambda succ: max_stack_per_scc.get(succ, 0))
            max_downstream_usage = max_stack_per_scc.get(best_next_scc, 0)

        max_stack_per_scc[scc_id] = (scc_stack_usage.get(scc_id, 0) +
                                     max_downstream_usage)
        scc_next_hop[scc_id] = best_next_scc

    results: SccResults = {}
    for scc_id, scc_nodes in enumerate(sccs):
        functions_in_scc = [{
            "name":
            function_info_map.get(addr, {}).get("label",
                                                f"unknown @ {hex(addr)}"),
            "address":
            addr,
            "stack_usage":
            function_info_map.get(addr, {}).get("stack_usage", 0)
        } for addr in scc_nodes]
        results[scc_id] = {
            "max_stack_usage": max_stack_per_scc.get(scc_id, 0),
            "functions": functions_in_scc,
            "next_scc_on_max_path": scc_next_hop.get(scc_id),
        }
    return results


def calculate_max_stack_no_cycles(
    entry_addr: int,
    cg: graph_helper.Graph,
    function_info_map: FunctionInfoByAddr,
) -> SccResults:
    """
    Calculates max stack usage assuming no recursion (breaking back-edges).
    Returns results in the same format as calculate_max_stack_usage (SccResults),
    treating each reachable node as an SCC of size 1.
    """
    memo: Dict[int, int] = {}
    next_hop: Dict[int, int] = {}
    path = set()

    # Increase recursion limit just in case
    sys.setrecursionlimit(5000)

    def dfs(u: int) -> int:
        if u in memo:
            return memo[u]
        if u in path:
            return 0  # Cycle detected, break it

        path.add(u)
        local_stack = function_info_map.get(u, {}).get("stack_usage", 0)
        max_downstream = 0
        best_v = None

        if u in cg.graph:
            for v in cg.graph[u]:
                if v in path:
                    continue  # Back-edge, ignore

                val = dfs(v)
                if val > max_downstream:
                    max_downstream = val
                    best_v = v

        total = local_stack + max_downstream
        path.remove(u)
        memo[u] = total
        if best_v is not None:
            next_hop[u] = best_v
        return total

    dfs(entry_addr)

    # Convert to SccResults format
    results: SccResults = {}
    # We use the address itself as the "SCC ID"
    for addr, max_stack in memo.items():
        results[addr] = {
            "max_stack_usage":
            max_stack,
            "functions": [{
                "name":
                function_info_map.get(addr, {}).get("label",
                                                    f"unknown @ {hex(addr)}"),
                "address":
                addr,
                "stack_usage":
                function_info_map.get(addr, {}).get("stack_usage", 0)
            }],
            "next_scc_on_max_path":
            next_hop.get(addr)
        }

    return results


def print_cycles(
        scc_results: SccResults,
        symbolizer_obj: Optional[symbolizer.Symbolizer] = None,
        demangler_obj: Optional[symbolizer.Demangler] = None) -> None:
    print("\n--- Detected Cycles (Recursive SCCs) ---")
    found_cycles = False
    for scc_id, res in scc_results.items():
        if len(res["functions"]) > 1:
            found_cycles = True
            print(
                f"SCC #{scc_id}: {len(res['functions'])} functions, Max Stack Contribution: {res['max_stack_usage']} bytes"
            )

            # Symbolize if available
            addrs = [f["address"] for f in res["functions"]]
            src_map = {}
            if symbolizer_obj:
                src_map = symbolizer_obj.symbolize(addrs)

            names = [f["name"] for f in res["functions"]]
            demangle_map = {}
            if demangler_obj:
                demangle_map = demangler_obj.demangle(names)

            for f in res["functions"]:
                addr = f["address"]
                name = f["name"]
                demangled_name = demangle_map.get(name, name)
                loc = src_map.get(addr, "")
                loc_str = f" [{loc}]" if loc else ""
                print(f"  - {demangled_name} (@{hex(addr)}){loc_str}")

    if not found_cycles:
        print("None")


def print_max_stack_path(
    start_scc_id: int,
    scc_results: SccResults,
    symbolizer_obj: Optional[symbolizer.Symbolizer] = None,
    demangler_obj: Optional[symbolizer.Demangler] = None,
) -> None:
    print("\n--- Path Contributing to Max Stack Usage ---")
    path_sccs = []
    current_scc_id: Optional[int] = start_scc_id

    visited_sccs = set()

    while current_scc_id is not None:
        if current_scc_id in visited_sccs:
            print(f"  ... (Cycle detected, stopping path trace)")
            break
        visited_sccs.add(current_scc_id)

        path_sccs.append(current_scc_id)
        current_scc_id = scc_results.get(current_scc_id,
                                         {}).get("next_scc_on_max_path")

    # Collect all addresses to symbolize and all names to demangle in batch
    all_addresses = []
    all_names = []
    for scc_id in path_sccs:
        for func in scc_results[scc_id]["functions"]:
            all_addresses.append(func["address"])
            all_names.append(func["name"])

    src_map = {}
    if symbolizer_obj and all_addresses:
        src_map = symbolizer_obj.symbolize(all_addresses)

    demangle_map = {}
    if demangler_obj and all_names:
        demangle_map = demangler_obj.demangle(all_names)

    for scc_id in path_sccs:
        scc_data = scc_results[scc_id]
        scc_functions = scc_data["functions"]

        if len(scc_functions) > 1:
            print(f"  - Beginning of cycle (SCC #{scc_id})")

        for func in scc_functions:
            addr = func["address"]
            usage = func.get("stack_usage", 0)
            name = func.get("name", f"<unknown_@{hex(addr)}>")
            demangled_name = demangle_map.get(name, name)
            loc = src_map.get(addr, "")
            loc_str = f" \n    -> {loc}" if loc else ""
            print(f"  - {demangled_name} (@{hex(addr)}): {usage} bytes{loc_str}")

        if len(scc_functions) > 1:
            print(f"  - End of cycle (SCC #{scc_id})")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate max stack usage from call graph and stack sizes."
    )
    parser.add_argument(
        "input_json",
        help=
        "Path to llvm-readelf --call-graph-info --stack-sizes --elf-output-style=JSON output",
    )
    parser.add_argument(
        "--entry-function",
        default="main",
        help="Entry function name (default: main)",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file (exclusions)",
    )
    parser.add_argument(
        "--llvm-symbolizer-path",
        help="Path to llvm-symbolizer binary for source location resolution",
    )
    parser.add_argument(
        "--llvm-cxxfilt-path",
        help="Path to llvm-cxxfilt binary for name demangling",
    )
    parser.add_argument(
        "--demangle",
        action="store_true",
        default=False,
        help="Demangle function names in the output (default: False)",
    )
    parser.add_argument(
        "--produce-intermediate",
        action="store_true",
        help="Produce intermediate results (type_map.json and consolidated_types.json) for investigation",
    )
    parser.add_argument(
        "--export-json",
        help="Path to export analysis data as JSON (for visualization)",
    )
    parser.add_argument(
        "--break-cycles",
        action="store_true",
        help=
        "Treat the graph as a DAG by ignoring back-edges (simulating no recursion).",
    )
    args = parser.parse_args()

    try:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error reading config file: {e}", file=sys.stderr)
            sys.exit(1)

    stack_sizes = call_graph.parse_stack_sizes(data)
    cg, name_to_addr, func_info = call_graph.build_call_graph(
        data, stack_sizes, config)

    if args.entry_function not in name_to_addr:
        print(f"Error: Entry function '{args.entry_function}' not found.",
              file=sys.stderr)
        sys.exit(1)

    entry_addr = name_to_addr[args.entry_function]

    # --- Analysis Logic ---
    results: SccResults = {}
    entry_scc_id = None
    max_stack = 0
    addr_to_scc_id = {}

    if args.break_cycles:
        # DAG Mode
        results = calculate_max_stack_no_cycles(entry_addr, cg, func_info)
        # In this mode, scc_id is the address itself
        if entry_addr in results:
            entry_scc_id = entry_addr
            max_stack = results[entry_scc_id]["max_stack_usage"]
        # Populate addr_to_scc_id for export logic (identity map)
        for addr in results:
            addr_to_scc_id[addr] = addr
    else:
        # SCC Mode (Standard)
        sccs = cg.get_sccs()
        for i, scc in enumerate(sccs):
            for addr in scc:
                addr_to_scc_id[addr] = i

        results = calculate_max_stack_usage(cg, func_info, sccs)
        if entry_addr in addr_to_scc_id:
            entry_scc_id = addr_to_scc_id[entry_addr]
            max_stack = results[entry_scc_id]["max_stack_usage"]

    # --- Reporting / Export ---

    # Initialize symbolizer if available
    sym = None
    if args.llvm_symbolizer_path and os.path.exists(args.llvm_symbolizer_path):
        elf_file = None
        for file_entry in data:
            f = file_entry.get("FileSummary", {}).get("File")
            if f:
                elf_file = f
                break

        if elf_file:
            # Handle relative paths for ELF
            if not os.path.exists(elf_file):
                json_dir = os.path.dirname(os.path.abspath(args.input_json))
                candidate = os.path.join(json_dir, elf_file)
                if os.path.exists(candidate):
                    elf_file = candidate

            if os.path.exists(elf_file):
                try:
                    sym = symbolizer.Symbolizer(args.llvm_symbolizer_path,
                                                elf_file)
                except Exception as e:
                    print(f"Warning: Failed to initialize symbolizer: {e}",
                          file=sys.stderr)
            else:
                print(
                    f"Warning: ELF file '{elf_file}' not found. Source info will be missing.",
                    file=sys.stderr)

    # Initialize demangler if available
    demangler = None
    if args.demangle and args.llvm_cxxfilt_path:
        if os.path.exists(args.llvm_cxxfilt_path):
            try:
                demangler = symbolizer.Demangler(args.llvm_cxxfilt_path)
            except Exception as e:
                print(f"Warning: Failed to initialize demangler: {e}",
                      file=sys.stderr)
        else:
            print(
                f"Warning: Demangler binary '{args.llvm_cxxfilt_path}' not found.",
                file=sys.stderr)

    if args.produce_intermediate:
        type_map = extract_type_map.extract_type_map(data, demangler)
        with open("type_map.json", 'w') as f:
            json.dump(type_map, f, indent=2)
        print("✅ Intermediate result 'type_map.json' produced.")

        consolidated = consolidate_sorted_table.consolidate_type_ids(
            data, type_map, demangler)
        with open("consolidated_types.json", 'w') as f:
            json.dump(consolidated, f, indent=2)
        print("✅ Intermediate result 'consolidated_types.json' produced.")

    if args.export_json:
        excluded_funcs = set()
        if config:
            exclusions = config.get("exclude_functions", [])
            if exclusions:
                first_item = exclusions[0]
                if isinstance(first_item, str):
                    excluded_funcs = set(exclusions)
                elif isinstance(first_item, dict):
                    excluded_funcs = set(
                        item.get("name") for item in exclusions
                        if "name" in item)

        type_id_to_targets = call_graph.extract_type_id_mapping(
            data, excluded_funcs)

        # Build Adjacency List for visualization
        adj_list = {}
        for file_entry in data:
            for entry in file_entry.get("CallGraph", []):
                func = entry.get("Function", {})
                caller_addr = func.get("Address")

                if caller_addr is None:
                    continue
                if any(name in excluded_funcs
                       for name in func.get("Names", [])):
                    continue

                if caller_addr not in adj_list:
                    adj_list[caller_addr] = {'direct': [], 'indirect': []}

                # Direct
                for callee in func.get("DirectCallees", []):
                    target = callee.get("Address")
                    if target is not None and target in func_info:
                        adj_list[caller_addr]['direct'].append(target)

                # Indirect
                for tid in func.get("IndirectTypeIDs", []):
                    targets = type_id_to_targets.get(tid, [])
                    valid_targets = [t for t in targets if t in func_info]
                    if valid_targets:
                        adj_list[caller_addr]['indirect'].append({
                            "typeId":
                            tid,
                            "targets":
                            valid_targets
                        })

        # Collect reachable nodes
        reachable = {}
        queue = [entry_addr]
        visited = {entry_addr}
        all_reachable_addrs = []

        while queue:
            u = queue.pop(0)
            all_reachable_addrs.append(u)

            info = func_info.get(u, {})
            cumulative_stack = 0

            scc_id = addr_to_scc_id.get(u)
            if scc_id is not None and scc_id in results:
                cumulative_stack = results[scc_id]["max_stack_usage"]

            node_data = {
                "name": info.get("label", f"@{hex(u)}"),
                "stack": info.get("stack_usage", 0),
                "max_cumulative_stack": cumulative_stack,
                "sccId": scc_id,
                "direct": [],
                "indirect": []
            }

            adj = adj_list.get(u, {'direct': [], 'indirect': []})

            for v in adj['direct']:
                node_data["direct"].append(v)
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

            for ind_group in adj['indirect']:
                node_data["indirect"].append(ind_group)
                for v in ind_group["targets"]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)

            reachable[u] = node_data

        if sym:
            print(f"Symbolizing {len(all_reachable_addrs)} functions...",
                  file=sys.stderr)
            src_map = sym.symbolize(all_reachable_addrs)
            for addr, src in src_map.items():
                if addr in reachable:
                    reachable[addr]["source"] = src

        if demangler:
            print(f"Demangling {len(all_reachable_addrs)} functions...",
                  file=sys.stderr)
            all_names = [reachable[addr]["name"] for addr in all_reachable_addrs]
            demangle_map = demangler.demangle(all_names)
            for addr in all_reachable_addrs:
                if addr in reachable:
                    name = reachable[addr]["name"]
                    reachable[addr]["demangled_name"] = demangle_map.get(
                        name, name)

        cycles_list = []
        if not args.break_cycles:
            for scc_id, res in results.items():
                funcs = res["functions"]
                if len(funcs) > 1:
                    cycles_list.append({
                        "id": scc_id,
                        "max_stack": res["max_stack_usage"],
                        "nodes": [f["address"] for f in funcs]
                    })

        export_data = {
            "entry_address": entry_addr,
            "entry_name": args.entry_function,
            "max_stack_usage": max_stack,
            "nodes": reachable,
            "cycles": cycles_list
        }

        with open(args.export_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Analysis data exported to {args.export_json}")

    else:
        # CLI Output
        if not args.break_cycles:
            print_cycles(results, sym, demangler)

        print(
            f"\n✅ Max stack usage from '{args.entry_function}': {max_stack} bytes"
        )

        # Unified print function
        print_max_stack_path(entry_scc_id, results, sym, demangler)

    if sym:
        sym.close()
    if demangler:
        demangler.close()


if __name__ == "__main__":
    main()
