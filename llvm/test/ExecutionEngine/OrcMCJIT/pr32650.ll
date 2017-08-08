; RUN: %lli -jit-kind=orc-mcjit %s

; This test is intended to verify that a function weakly defined in
; JITted code, and strongly defined in the main executable, can be
; correctly resolved when called from elsewhere in JITted code.

; This test makes the assumption that the lli executable in compiled
; to export symbols (e.g. --export-dynamic), and that is actually does
; contain the symbol LLVMInitializeCodeGen.  (Note that this function
; is not actually called by the test.  The test simply verifes that
; the reference can be resolved without relocation errors.)

define linkonce_odr void @LLVMInitializeCodeGen() {
entry:
  ret void
}

define void @test() {
entry:
  call void @LLVMInitializeCodeGen()
  ret void
}

define i32 @main() {
entry:
  ret i32 0
}

