# .callgraph Section Layout

The `.callgraph` section is used to store call graph information for each function, which can be used for post-link analyses and optimizations. The section contains a series of records, with each record corresponding to a single function.

## Per Function Record Layout

Each record in the `.callgraph` section has the following binary layout:

| Field                        | Type          | Size (bits) | Description                                                                                             |
| ---------------------------- | ------------- | ----------- | ------------------------------------------------------------------------------------------------------- |
| Format Version               | `uint32_t`    | 32          | The version of the record format. The current version is 0.                                             |
| Function Entry PC            | `uintptr_t`   | 32/64       | The address of the function's entry point.                                                              |
| Function Kind                | `uint8_t`     | 8           | An enum indicating the function's properties (e.g., if it's an indirect call target).                   |
| Function Type ID             | `uint64_t`    | 64          | The type ID of the function. This field is **only** present if `Function Kind` is `INDIRECT_TARGET_KNOWN_TID`. |
| Number of Indirect Callsites | `uint32_t`    | 32          | The number of indirect call sites within the function.                                                  |
| Indirect Callsites Array     | `Callsite[]`  | Variable    | An array of `Callsite` records, with a length of `Number of Indirect Callsites`.                        |


### Indirect Callsite Record Layout

Each record in the `Indirect Callsites Array` has the following layout:

| Field             | Type        | Size (bits) | Description                               |
| ----------------- | ----------- | ----------- | ----------------------------------------- |
| Type ID           | `uint64_t`  | 64          | The type ID of the indirect call target.  |
| Callsite PC       | `uintptr_t` | 32/64       | The address of the indirect call site.    |
