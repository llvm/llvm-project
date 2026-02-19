;; When a libcall was not brought into the link, it can be used iff it is
;; defined in native code, not bitcode.
; RUN: opt %s -o %t -module-summary -mtriple x86_64-unknown-linux-musl
; RUN: llvm-lto2 run -o %t.bitcode \
; RUN:   -r %t,foo,plx \
; RUN:   -r %t,memcmp,x \
; RUN:   -r %t,bcmp,pl --bitcode-libfuncs=bcmp %t -save-temps
; RUN: llvm-dis %t.bitcode.1.4.opt.bc -o - | FileCheck --check-prefixes=CHECK,BITCODE %s
; RUN: llvm-lto2 run -o %t.native \
; RUN:   -r %t,foo,plx \
; RUN:   -r %t,memcmp,x \
; RUN:   -r %t,bcmp,pl %t -save-temps
; RUN: llvm-dis %t.native.1.4.opt.bc -o - | FileCheck --check-prefixes=CHECK,NATIVE %s
define i1 @foo(ptr %0, ptr %1, i64 %2) {
  ; CHECK-LABEL: define{{.*}}i1 @foo
  ; BITCODE-NEXT: %cmp = {{.*}}call i32 @memcmp
  ; BITCODE-NEXT: %eq = icmp eq i32 %cmp, 0
  ; NATIVE-NEXT: %bcmp = {{.*}}call i32 @bcmp
  ; NATIVE-NEXT: %eq = icmp eq i32 %bcmp, 0
  ; CHECK-NEXT: ret i1 %eq


  %cmp = call i32 @memcmp(ptr %0, ptr %1, i64 %2)
  %eq = icmp eq i32 %cmp, 0
  ret i1 %eq
}

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @bcmp(ptr, ptr, i64)
