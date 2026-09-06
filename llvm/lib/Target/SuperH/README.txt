//===---------------------------------------------------------------------===//
// Ideas for the backend.
//===---------------------------------------------------------------------===//

  * Improve SuperHFillDelaySlots pass to fill delay slots from branch
    destinations.

  * Add more variations to MOV*SFR and MOV*SFR for large stack sizes.
  * Improve compactness of constant islands.

//===---------------------------------------------------------------------===//
// Known Issues
//===---------------------------------------------------------------------===//

  * Only SH2 level codegen is implemented.
  * O1 compilation fails due to misgenerated fallthrough MBBs
  * 64-bit integers are not handled.
  * Varargs are not implemented.