;; Verify that byte types are encoded as integer types in DXIL bitcode.
; RUN: llc --filetype=obj %s --stop-after=dxil-write-bitcode -o %t
; RUN: llvm-bcanalyzer --dump %t | FileCheck %s

target triple = "dxil--shadermodel6.5-library"

;; Since bitcode defines types up front and then refers back to them using
;; their indices, we unfortunately need to hardcode the entire type block here,
;; which means this test may be brittle.

;; See https://llvm.org/docs/BitCodeFormat.html for help interpreting below.
;
; CHECK:      <TYPE_BLOCK_ID
; CHECK-NEXT:   <NUMENTRY op0=24/>
; CHECK-NEXT:   <INTEGER op0=8/>
; CHECK-NEXT:   <POINTER {{.*}} op0=0 op1=0/>
;; 2: STRUCT_NAME + OPAQUE only take up one index here.
; CHECK-NEXT:   <STRUCT_NAME {{.*}}/> record string = 'dxilOpaquePtrReservedName'
; CHECK-NEXT:   <OPAQUE op0=0/>
;; 3: i8
; CHECK-NEXT:   <INTEGER op0=8/>
;; 4: [32 x i8]
; CHECK-NEXT:   <ARRAY {{.*}} op0=32 op1=3/>
;; 5-12: The return and operand types of @bytes, except `i8`,
; CHECK-NEXT:   <VOID/>
; CHECK-NEXT:   <INTEGER op0=1/>
; CHECK-NEXT:   <INTEGER op0=3/>
; CHECK-NEXT:   <INTEGER op0=5/>
; CHECK-NEXT:   <INTEGER op0=16/>
; CHECK-NEXT:   <INTEGER op0=32/>
; CHECK-NEXT:   <INTEGER op0=64/>
; CHECK-NEXT:   <INTEGER op0=128/>
;; 13: <8 x i5>
; CHECK-NEXT:   <VECTOR op0=8 op1=8/>
;; 14: <2 x i64>
; CHECK-NEXT:   <VECTOR op0=2 op1=11/>
;; 15: void(i1, i3, i5, i8, i16, i32, i64, i128, <8 x i5>, <2 x i64>)
; CHECK-NEXT:   <FUNCTION {{.*}} op0=0 op1=5 op2=6 op3=7 op4=8 op5=3 op6=9 op7=10 op8=11 op9=12 op10=13 op11=14/>
; CHECK-NEXT:   <POINTER {{.*}} op0=15 op1=0/>
;; 17: b32()
; CHECK-NEXT:   <FUNCTION {{.*}} op0=0 op1=10/>
; CHECK-NEXT:   <POINTER {{.*}} op0=17 op1=0/>
;; 19: b128()
; CHECK-NEXT:   <FUNCTION {{.*}} op0=0 op1=12/>
; CHECK-NEXT:   <POINTER {{.*}} op0=19 op1=0/>
; CHECK-NEXT:   <POINTER {{.*}} op0=4 op1=0/>
; CHECK-NEXT:   <METADATA/>
; CHECK-NEXT:   <INTEGER op0=32/>
; CHECK-NEXT: </TYPE_BLOCK_ID>

;; Sanity check that the globals are coherently ordered.
; CHECK:      <GLOBALVAR {{.*}} op0=4
; CHECK-NEXT: <FUNCTION op0=15
; CHECK-NEXT: <FUNCTION op0=17
; CHECK-NEXT: <FUNCTION op0=19

; CHECK:      <VALUE_SYMTAB
; CHECK-NEXT:   <ENTRY {{.*}} op0=0 {{.*}}/> record string = 'a'
; CHECK-NEXT:   <ENTRY {{.*}} op0=1 {{.*}}/> record string = 'bytes'
; CHECK-NEXT:   <ENTRY {{.*}} op0=3 {{.*}}/> record string = 'constant128'
; CHECK-NEXT:   <ENTRY {{.*}} op0=2 {{.*}}/> record string = 'constant32'
; CHECK-NEXT: </VALUE_SYMTAB>

@a = common global [32 x b8] zeroinitializer, align 1

define void @bytes(b1 %a, b3 %b, b5 %c, b8 %d, b16 %e, b32 %f, b64 %g, b128 %h, <8 x b5> %i, <2 x b64> %j) {
  ; CHECK: <FUNCTION_BLOCK
  ret void
}

define b32 @constant32() {
  ; CHECK: <FUNCTION_BLOCK
  ; CHECK:      <CONSTANTS_BLOCK
  ; CHECK-NEXT:   <SETTYPE {{.*}} op0=10/>
  ;; The value here is signed variable-length encoded, so the value is doubled
  ; CHECK-NEXT:   <INTEGER {{.*}} op0=246/>
  ret b32 123
}

define b128 @constant128() {
  ; CHECK: <FUNCTION_BLOCK
  ; CHECK:      <CONSTANTS_BLOCK
  ; CHECK-NEXT:   <SETTYPE {{.*}} op0=12/>
  ;; The value here is signed variable-length encoded, so the value is doubled
  ; CHECK-NEXT:   <WIDE_INTEGER op0=24682468/>
  ret b128 12341234
}
