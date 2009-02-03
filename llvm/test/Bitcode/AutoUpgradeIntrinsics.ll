; This isn't really an assembly file. It just runs test on bitcode to ensure
; it is auto-upgraded.
; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.ct}
; RUN: llvm-dis < %s.bc | \
; RUN:   not grep {llvm\\.part\\.set\\.i\[0-9\]*\\.i\[0-9\]*\\.i\[0-9\]*}
; RUN: llvm-dis < %s.bc | \
; RUN:   not grep {llvm\\.part\\.select\\.i\[0-9\]*\\.i\[0-9\]*}
; RUN: llvm-dis < %s.bc | \
; RUN:   not grep {llvm\\.bswap\\.i\[0-9\]*\\.i\[0-9\]*}

