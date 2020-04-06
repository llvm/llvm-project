; REQUIRES: x86_64-apple, x86-registered-target, arm-registered-target
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -filetype=obj -mtriple=x86_64-apple-macosx -o %t/foo.x86_64.o \
; RUN:   %p/Inputs/foo-return-i32-0.ll
; RUN: llvm-ar r %t/foo.x86_64.a %t/foo.x86_64.o
; RUN: llc -filetype=obj -mtriple=arm-apple-ios -o %t/foo.arm.o \
; RUN:   %p/Inputs/foo-return-i32-0.ll
; RUN: llvm-ar r %t/foo.arm.a %t/foo.arm.o
; RUN: llvm-lipo -create %t/foo.x86_64.a %t/foo.arm.a -output %t/foo.a
; RUN: lli -jit-kind=orc-lazy -extra-archive %t/foo.a %s
;
; Check that MachO universal binaries containing archives work.
; This test compiles two copies of a simple int foo() function that returns
; zero, one copy for x86_64 and one for arm. It then puts each of these in an
; archive and combines these two archives into a macho universal binary.
; Finally we execute a main function that references foo to ensure that the
; x86-64 copy is correctly found and linked.

declare i32 @foo()

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
entry:
  %0 = call i32 @foo()
  ret i32 %0
}
