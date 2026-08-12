#!/usr/bin/env python3

import json
import collections
import sys
import os
import argparse


def consolidate_type_ids(call_graph_data, type_map_data, demangler=None):
    # 1. Find IndirectTargetTypeIDs that are part of only one function's indirect target list.
    type_id_to_functions = collections.defaultdict(set)
    addr_to_name = {}
    all_names = set()

    if isinstance(call_graph_data, dict):
        call_graph_data = [call_graph_data]

    for entry in call_graph_data:
        call_graph = entry.get('CallGraph', [])
        for item in call_graph:
            func = item.get('Function', {})
            func_id = func.get('Address')  # Use Address as unique identifier
            if func_id is None:
                continue

            names = func.get('Names', [])
            func_name = names[0] if names else f"Unknown_Addr_{func_id}"
            addr_to_name[func_id] = func_name
            all_names.update(names)

            indirect_ids = func.get('IndirectTypeIDs', [])
            for tid in indirect_ids:
                type_id_to_functions[tid].add(func_id)

    demangle_map = {}
    if demangler and all_names:
        demangle_map = demangler.demangle(list(all_names))

    unique_ids = []
    for tid, func_set in type_id_to_functions.items():
        if len(func_set) == 1:
            unique_ids.append(tid)

    # 2. For these IDs, count number of functions in type_map.json
    results = []

    for tid in unique_ids:
        tid_str = str(tid)
        targets = type_map_data.get(tid_str, [])

        # Get the single function address
        func_addr = list(type_id_to_functions[tid])[0]
        mangled_func_name = addr_to_name.get(func_addr, "Unknown")
        func_name = demangle_map.get(mangled_func_name, mangled_func_name)

        results.append({
            "type_id": tid,
            "target_count": len(targets),
            "calling_function": func_name
        })

    # 3. Sort by count descending
    results.sort(key=lambda x: x["target_count"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate and sort indirect target types.")
    parser.add_argument("--call-graph",
                        default='call_graph_out.json',
                        help="Path to call graph JSON")
    parser.add_argument("--type-map",
                        default='type_map.json',
                        help="Path to type map JSON")
    parser.add_argument("--output",
                        "-o",
                        default='consolidated_types.json',
                        help="Output JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.call_graph):
        print(f"Error: {args.call_graph} not found.")
        sys.exit(1)
    if not os.path.exists(args.type_map):
        print(f"Error: {args.type_map} not found.")
        sys.exit(1)

    try:
        with open(args.call_graph, 'r') as f:
            call_graph_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {args.call_graph}.")
        sys.exit(1)

    try:
        with open(args.type_map, 'r') as f:
            type_map_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {args.type_map}.")
        sys.exit(1)

    results = consolidate_type_ids(call_graph_data, type_map_data)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Consolidated types written to {args.output}")


if __name__ == '__main__':
    main()
