; RUN: %python %S/Inputs/allocsite-removable-timeout-gen.py 12000 | \
; RUN:   opt -passes=instcombine \
; RUN:       -instcombine-max-allocsite-removable-users=128 \
; RUN:       -disable-output

; Compile-time regression test for isAllocSiteRemovable().
; The generated function contains an alloca with many direct users plus one
; escaping use, so the alloca is not removable. Without the bailout, this
; takes O(N^2) time in the user walk. Success = InstCombine finishes quickly.
