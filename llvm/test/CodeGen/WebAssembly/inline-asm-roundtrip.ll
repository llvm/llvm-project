; RUN: llc < %s | llvm-mc -triple=wasm32-unknown-unknown | FileCheck --match-full-lines %s

; Test basic inline assembly can actually be assembled by the assembler.

; .ll code below is the result of this code run thru
; clang -target wasm32-unknown-unknown-wasm -O2 -S -emit-llvm test.c

; int main(int argc, const char *argv[]) {
;   int src = 1;
;   int dst;
;   asm ("i32.const\t2\n"
;        "\tlocal.get\t%1\n"
;        "\ti32.add\n"
;        "\tlocal.set\t%0"
;        : "=r" (dst)
;        : "r" (src));
;   return dst != 3;
; }

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: main:
; CHECK-NEXT:	.functype main (i32, i32) -> (i32)
; CHECK-NEXT:	.local  	i32
; CHECK-NEXT:	i32.const	1
; CHECK-NEXT:	local.set	[[SRC:[0-9]+]]
; CHECK-NEXT:	i32.const	2
; CHECK-NEXT:	local.get	[[SRC]]
; CHECK-NEXT:	i32.add
; CHECK-NEXT:	local.set	[[DST:[0-9]+]]
; CHECK-NEXT:	local.get	[[DST]]
; CHECK-NEXT:	i32.const	3
; CHECK-NEXT:	i32.ne

define i32 @main(i32 %argc, i8** nocapture readnone %argv) #0 {
entry:
  %0 = tail call i32 asm "i32.const\092\0A\09local.get\09$1\0A\09i32.add\0A\09local.set\09$0", "=r,r"(i32 1) #1
  %cmp = icmp ne i32 %0, 3
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
