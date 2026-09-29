;; When we compile object with "-flto", the info of "-mnan=2008"
;; and "-mfp32/-mfpxx/-mfp64" will be missing in the result IR file.
;; Thus the asm/obj files will have wrong format.
;; With D140270 we extract these info from the first function,
;; and set it for the whole compile unit.
; RUN: llc %s -o - | FileCheck %s

target triple = "mipsel-unknown-linux-gnu"

define dso_local void @test() #0 {
  ret void
}

attributes #0 = { "target-cpu"="mips32r2" "target-features"="+fp64,+mips32r2,+nan2008" }


; CHECK: .nan    2008
; CHECK: .module fp=64
