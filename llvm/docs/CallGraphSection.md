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

Each function and direct callee is identified by exactly one of `Address` or `Relocation`. If a relocation applies to its address field, as in relocatable object files, `Relocation` describes that relocation. Otherwise, `Address` holds the value stored in the field. Either one may be accompanied by `Names`, the symbolization of that address for human readers.

### Function Object

| Field                      | Type             | Description |
| -------------------------- | ---------------- | ----------- |
| `Names`                    | array of strings | The symbolization of the function's entry address, similar to what `llvm-symbolizer` shows: the names of the function symbols defined at that address. When `Relocation` is present, the address is where the relocation points: the symbol's value plus the addend, within the symbol's section. For human readers only, so it can differ from the relocation's `SymbolName`, for example when the relocation uses a section symbol. Omitted if there are none, for example when the relocation's symbol is undefined. |
| `Address`                  | integer          | The value stored in the function entry PC field, which in executables and shared objects is the function's entry address. On ARM, the Thumb bit is cleared. Printed when no relocation applies to the field. |
| `Relocation`               | object           | The relocation that applies to the function entry PC field. See below. Printed instead of `Address`. |
| `Version`                  | integer          | The format version. |
| `IsIndirectTarget`         | boolean          | Whether the function is a potential indirect call target. |
| `TypeID`                   | integer          | The function type ID. |
| `NumDirectCallees`         | integer          | The number of entries in `DirectCallees`. |
| `DirectCallees`            | array of objects | One object per direct callee, with the same `Names`, `Address` and `Relocation` fields as the function. Callees identified by `Address` are listed once per address. |
| `NumIndirectTargetTypeIDs` | integer          | The number of entries in `IndirectTypeIDs`. |
| `IndirectTypeIDs`          | array of integers | The indirect call target type IDs. |

### Relocation Object

| Field         | Type    | Description |
| ------------- | ------- | ----------- |
| `Type`        | object  | The relocation type, with its `Name` (such as `R_X86_64_64`) and numeric `Value`. |
| `SymbolIndex` | integer | The symbol table index of the relocation's symbol, or 0 if it has none. |
| `SymbolName`  | string  | The symbol's name exactly as given by `st_name`. Omitted if the symbol has no name, such as an `STT_SECTION` symbol. |
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

A function in a relocatable object file that calls a local function `foo`, a global function `bar` and an undefined function `ext`. The relocation for `foo` uses the `.text` section symbol, which has no name, so it has no `SymbolName`, but `Names` still shows `foo`. `ext` is not defined in the file, so it has no `Names`:

```json
"CallGraph": [
  {
    "Function": {
      "Names": [
        "caller"
      ],
      "Relocation": {
        "Type": {
          "Name": "R_X86_64_64",
          "Value": 1
        },
        "SymbolIndex": 6,
        "SymbolName": "caller"
      },
      "Version": 0,
      "IsIndirectTarget": true,
      "TypeID": 9080559750644022485,
      "NumDirectCallees": 3,
      "DirectCallees": [
        {
          "Names": [
            "foo"
          ],
          "Relocation": {
            "Type": {
              "Name": "R_X86_64_64",
              "Value": 1
            },
            "SymbolIndex": 2,
            "Addend": 48
          }
        },
        {
          "Names": [
            "bar"
          ],
          "Relocation": {
            "Type": {
              "Name": "R_X86_64_64",
              "Value": 1
            },
            "SymbolIndex": 4,
            "SymbolName": "bar"
          }
        },
        {
          "Relocation": {
            "Type": {
              "Name": "R_X86_64_64",
              "Value": 1
            },
            "SymbolIndex": 7,
            "SymbolName": "ext"
          }
        }
      ],
      "NumIndirectTargetTypeIDs": 0,
      "IndirectTypeIDs": []
    }
  }
]
```
