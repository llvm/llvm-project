; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck %s

; unsigned int foo(void) {
;   return 0;
; }
;
; int main() {
; L: __attribute__ ((unused));
;   static const unsigned int arr[] =
;   {
;     (unsigned int) &&x  - (unsigned int)&&L ,
;     (unsigned int) &&y  - (unsigned int)&&L
;   };
;
;   unsigned int ret = foo();
;   void* g = (void *) ((unsigned int)&&L + arr[ret]);
;   goto *g;
;
; x:
;   return 15;
; y:
;   return 25;
; }

define i32 @foo() local_unnamed_addr {
entry:
  ret i32 0
}

define i32 @main() {
entry:
  br label %L

L:                                                ; preds = %L, %entry
  indirectbr i8* inttoptr (i32 add (i32 ptrtoint (i8* blockaddress(@main, %L) to i32), i32 sub (i32 ptrtoint (i8* blockaddress(@main, %return) to i32), i32 ptrtoint (i8* blockaddress(@main, %L) to i32))) to i8*), [label %return, label %L]

return:                                           ; preds = %L
  ret i32 15
}


; CHECK:     lwz 3, .LC0-.LTOC(30)
; CHECK-NOT: li 3, .Ltmp1-.L1$pb@l
; CHECK-NOT: addis 4, 30, .Ltmp1-.L1$pb@ha