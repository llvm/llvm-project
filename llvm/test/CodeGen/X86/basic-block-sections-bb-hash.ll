; BB section test with basic block hashes.
;
; RUN: llc %s -O0 -mtriple=x86_64-pc-linux -function-sections -filetype=obj -basic-block-address-map -emit-bb-hash -o %t.o
; RUN: obj2yaml %t.o -o %t.yaml
;
;; Profile for version 1:
; RUN: echo 'v1' > %t
; RUN: echo 'f foo' >> %t
; RUN: echo 'g 0:10,1:9,2:1 1:8,3:8 2:2,3:2 3:11' >> %t
; RUN: echo 'c 0 2 3' >> %t

; These commands read BB hashes from SHT_LLVM_BB_ADDR_MAP 
; and put them into the basic blocks sections profile.
; RUN: grep -E '^\s+(- ID:|Hash:)' %t.yaml | \
; RUN: grep -B1 'Hash:' | \
; RUN: sed 's/^\s*//; s/^- ID: *//; s/Hash: *0x//' | \
; RUN: paste -d: - - | \
; RUN: tr '\n' ' ' | \
; RUN: sed 's/ $/\n/; s/^/h /' >> %t
;
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t | FileCheck %s
;
define void @foo(i1 zeroext) nounwind {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, ptr %2, align 1
  %4 = load i8, ptr %2, align 1
  %5 = trunc i8 %4 to i1
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = call i32 @bar()
  br label %10

8:                                                ; preds = %1
  %9 = call i32 @baz()
  br label %10

10:                                               ; preds = %8, %6
  ret void
}

declare i32 @bar() #1

declare i32 @baz() #1

; CHECK: .section	.text.foo,"ax",@progbits
; CHECK: callq baz
; CHECK: retq
; CHECK: .section	.text.split.foo,"ax",@progbits
; CHECK: callq bar
