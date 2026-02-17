#!/usr/bin/env python3

import argparse
import json
import sys
import call_graph


def main():
    parser = argparse.ArgumentParser(
        description="Extract TypeID to Function Name mapping.")
    parser.add_argument("input_json", help="Path to llvm-readelf JSON output")
    parser.add_argument("--output",
                        "-o",
                        default="type_map.json",
                        help="Output JSON file")
    args = parser.parse_args()

    try:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    # We don't need stack sizes for this
    type_id_to_addrs = call_graph.extract_type_id_mapping(data)

    # We need a map from Address -> Names to print readable output
    addr_to_names = {}
    for file_entry in data:
        for entry in file_entry.get("CallGraph", []):
            func = entry.get("Function", {})
            addr = func.get("Address")
            names = func.get("Names", [])
            if addr is not None and names:
                addr_to_names[addr] = names

    # Build the final map: TypeID (str) -> List[FuncName]
    type_map = {}
    for type_id, addrs in type_id_to_addrs.items():
        # Use str(type_id) because JSON keys must be strings
        func_names = []
        for addr in addrs:
            func_names.extend(
                addr_to_names.get(addr, [f"<unknown_@{hex(addr)}>"]))
        type_map[str(type_id)] = sorted(list(set(func_names)))

    with open(args.output, 'w') as f:
        json.dump(type_map, f, indent=2)

    print(f"✅ Type map written to {args.output}")


if __name__ == "__main__":
    main()
