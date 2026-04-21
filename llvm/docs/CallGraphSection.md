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
