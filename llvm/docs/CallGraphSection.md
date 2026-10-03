# .llvm.callgraph Section Layout

The `.llvm.callgraph` section is used to store call graph information for each function. The section contains a series of records, with each record corresponding to a single function.

## Per Function Record Layout

Each record in the `.llvm.callgraph` section has the following binary layout:

| Field                                  | Type          | Size (bits) | Description                                                                                             |
| -------------------------------------- | ------------- | ----------- | ------------------------------------------------------------------------------------------------------- |
| Format Version                         | `uint8_t`     | 8           | The version of the record format. The current version is 0.                                             |
| Flags                                  | `uint8_t`     | 8           | A bitfield where: Bit 0 is set if the function is a potential indirect call target; Bit 1 is set if there are direct callees; Bit 2 is set if there are indirect callees. The remaining 5 bits are reserved. |
| Function Entry PC                      | `uintptr_t`   | 32/64       | The address of the function's entry point.                                                              |
| Function Type ID                       | `uint64_t`    | 64          | The type ID of the function. This field is non-zero if the function is a potential indirect call target and its type is known. |
| Number of Unique Direct Callees        | `ULEB128`     | Variable    | The number of unique direct call destinations from this function. This field is only present if there is at least one direct callee. |
| Direct Callees Array                   | `uintptr_t[]` | Variable    | An array of unique direct callee entry point addresses. This field is only present if there is at least one direct callee. |
| Number of Unique Indirect Target Type IDs| `ULEB128`     | Variable    | The number of unique indirect call target type IDs. This field is only present if there is at least one indirect target type ID. |
| Indirect Target Type IDs Array         | `uint64_t[]`  | Variable    | An array of unique indirect call target type IDs. This field is only present if there is at least one indirect target type ID. |

## Dumping with llvm-readobj

`llvm-readobj --call-graph-info` prints the records of every `SHT_LLVM_CALL_GRAPH` section in a file. With `--elf-output-style=JSON`, the output is a `CallGraph` array that holds one `Function` object per record. The default `LLVM` output style prints the same fields, with addresses and type IDs in hexadecimal.

In executables and shared objects, functions are identified by their addresses. In relocatable object files the address fields are not final, so each function is identified by the relocation that applies to its address field instead.

### Function Object

| Field                      | Type             | Description |
| -------------------------- | ---------------- | ----------- |
| `Names`                    | array of strings | The names of the function symbols at `Address`. Omitted if there are none. Executables and shared objects only. |
| `Address`                  | integer          | The function entry address. On ARM, the Thumb bit is cleared. Executables and shared objects only. |
| `Reloc`                    | object           | The relocation that applies to the function entry PC field. See below. Relocatable object files only. |
| `Offset`                   | integer          | The offset of the function entry PC field within the section. Printed instead of `Reloc` when the relocation cannot be determined. Relocatable object files only. |
| `Version`                  | integer          | The format version. |
| `IsIndirectTarget`         | boolean          | Whether the function is a potential indirect call target. |
| `TypeID`                   | integer          | The function type ID. |
| `NumDirectCallees`         | integer          | The number of entries in `DirectCallees`. |
| `DirectCallees`            | array of objects | One object per direct callee, identified the same way as the function: by `Names` and `Address`, or by `Reloc` or `Offset`. In executables and shared objects, each callee address is listed once. |
| `NumIndirectTargetTypeIDs` | integer          | The number of entries in `IndirectTypeIDs`. |
| `IndirectTypeIDs`          | array of integers | The indirect call target type IDs. |

### Reloc Object

| Field         | Type    | Description |
| ------------- | ------- | ----------- |
| `SymbolIndex` | integer | The symbol table index of the relocation's symbol, or 0 if it has none. |
| `SymbolName`  | string  | The name that the symbol's `st_name` refers to, demangled if `--demangle` is given. Omitted if the symbol has no name, such as an `STT_SECTION` symbol. |
| `Addend`      | integer | The relocation addend. For relocations without an explicit addend, such as `SHT_REL` relocations, this is the value stored in the field. Omitted if zero. |

### Examples

A function in a shared object that calls `foo` and `bar`:

```json
"CallGraph": [
  {
    "Function": {
      "Names": [
        "main"
      ],
      "Address": 6080,
      "Version": 0,
      "IsIndirectTarget": false,
      "TypeID": 0,
      "NumDirectCallees": 2,
      "DirectCallees": [
        {
          "Names": [
            "foo"
          ],
          "Address": 6032
        },
        {
          "Names": [
            "bar"
          ],
          "Address": 6048
        }
      ],
      "NumIndirectTargetTypeIDs": 0,
      "IndirectTypeIDs": []
    }
  }
]
```

A function in a relocatable object file that calls two local functions. The callees are referenced through the section symbol, so they have no `SymbolName`, and the addends give their offsets within the section:

```json
"CallGraph": [
  {
    "Function": {
      "Reloc": {
        "SymbolIndex": 3,
        "SymbolName": "caller"
      },
      "Version": 0,
      "IsIndirectTarget": true,
      "TypeID": 9080559750644022485,
      "NumDirectCallees": 2,
      "DirectCallees": [
        {
          "Reloc": {
            "SymbolIndex": 1,
            "Addend": 16
          }
        },
        {
          "Reloc": {
            "SymbolIndex": 1,
            "Addend": 24
          }
        }
      ],
      "NumIndirectTargetTypeIDs": 0,
      "IndirectTypeIDs": []
    }
  }
]
```
