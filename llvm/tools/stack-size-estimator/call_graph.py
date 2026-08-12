import json
import sys
from typing import Dict, List, Tuple, Any, Optional
import graph_helper

# Type aliases
JsonData = Dict[str, Any]
StackSizeMap = Dict[str, int]
Address = int
FuncName = str
FunctionInfoByAddr = Dict[Address, Dict[str, Any]]


def extract_type_id_mapping(
    data: List[JsonData], excluded_funcs: set[str] = set()
) -> Dict[int, List[Address]]:
    """
    Extracts a mapping from TypeID to a list of function addresses.
    """
    type_id_to_targets: Dict[int, List[Address]] = {}
    for file_entry in data:
        for entry in file_entry.get("CallGraph", []):
            func = entry.get("Function", {})
            addr = func.get("Address")
            names = func.get("Names", [])
            type_id = func.get("TypeID")

            if addr is None:
                continue

            # Check exclusion
            if any(name in excluded_funcs for name in names):
                continue

            if type_id is not None:
                if type_id not in type_id_to_targets:
                    type_id_to_targets[type_id] = []
                type_id_to_targets[type_id].append(addr)
    return type_id_to_targets


def build_call_graph(
    data: List[JsonData],
    stack_sizes: StackSizeMap,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[graph_helper.Graph, Dict[FuncName, Address], FunctionInfoByAddr]:
    """
    Constructs a directed graph from call graph metadata and stack sizes.
    """
    call_graph = graph_helper.Graph()
    fn_name_to_addr: Dict[FuncName, Address] = {}
    function_info_map: FunctionInfoByAddr = {}

    excluded_funcs = set()
    if config:
        exclusions = config.get("exclude_functions", [])
        if exclusions:
            first_item = exclusions[0]
            if isinstance(first_item, str):
                excluded_funcs = set(exclusions)
            elif isinstance(first_item, dict):
                excluded_funcs = set(
                    item.get("name") for item in exclusions if "name" in item)

    # Map TypeID -> List[Address]
    type_id_to_targets = extract_type_id_mapping(data, excluded_funcs)

    # First pass: Register all functions
    for file_entry in data:
        for entry in file_entry.get("CallGraph", []):
            func = entry.get("Function", {})
            addr = func.get("Address")
            names = func.get("Names", [])

            if addr is None:
                continue

            # Use the first name as the primary label
            primary_name = names[0] if names else f"<unknown_@{hex(addr)}>"

            # Check exclusion
            if any(name in excluded_funcs for name in names):
                continue

            for name in names:
                fn_name_to_addr[name] = addr

            # Get stack usage
            stack_usage = 0
            for name in names:
                if name in stack_sizes:
                    stack_usage = stack_sizes[name]
                    break

            call_graph.add_vertex(addr)
            function_info_map[addr] = {
                "stack_usage": stack_usage,
                "label": primary_name,
                "names": names
            }

    # Second pass: Add edges
    for file_entry in data:
        for entry in file_entry.get("CallGraph", []):
            func = entry.get("Function", {})
            caller_addr = func.get("Address")
            names = func.get("Names", [])

            if caller_addr is None:
                continue

            if any(name in excluded_funcs for name in names):
                continue

            # Direct calls
            for callee in func.get("DirectCallees", []):
                callee_addr = callee.get("Address")
                if callee_addr is not None:
                    # Only add edge if callee exists in graph (might be excluded)
                    if callee_addr in function_info_map:
                        call_graph.add_edge(caller_addr, callee_addr)

            # Indirect calls (Type IDs)
            for indirect_type_id in func.get("IndirectTypeIDs", []):
                if indirect_type_id in type_id_to_targets:
                    for target_addr in type_id_to_targets[indirect_type_id]:
                        if target_addr in function_info_map:
                            call_graph.add_edge(caller_addr, target_addr)

    return call_graph, fn_name_to_addr, function_info_map


def parse_stack_sizes(stack_data: List[JsonData]) -> StackSizeMap:
    """
    Parses the stack size JSON data into a simple dictionary.
    """
    sizes: StackSizeMap = {}
    # The format is typically a list of file summaries, each containing "StackSizes"
    # or just a list of stack sizes depending on how it's concatenated.
    # llvm-readelf --stack-sizes --elf-output-style=JSON output:
    # [ { "StackSizes": [ { "Entry": { "Functions": [...], "Size": 10 } } ] } ]

    for file_summary in stack_data:
        for entry_item in file_summary.get("StackSizes", []):
            entry = entry_item.get("Entry", {})
            size = entry.get("Size", 0)
            for func_name in entry.get("Functions", []):
                sizes[func_name] = size
    return sizes
