; RUN: llc < %s -march=avr | FileCheck %s

; CHECK-LABEL: ret_void_args_struct_i8_i32
define void @ret_void_args_struct_i8_i32({ i8, i32 } %a) {
start:
  ; CHECK:      sts     4, r20
  %0 = extractvalue { i8, i32 } %a, 0
  store volatile i8 %0, i8* inttoptr (i64 4 to i8*)

  ; CHECK-NEXT: sts     8, r24
  ; CHECK-NEXT: sts     7, r23
  ; CHECK-NEXT: sts     6, r22
  ; CHECK-NEXT: sts     5, r21
  %1 = extractvalue { i8, i32 } %a, 1
  store volatile i32 %1, i32* inttoptr (i64 5 to i32*)
  ret void
}

; CHECK-LABEL: ret_void_args_struct_i8_i8_i8_i8
define void @ret_void_args_struct_i8_i8_i8_i8({ i8, i8, i8, i8 } %a) {
start:
  ; CHECK:      sts     4, r22
  %0 = extractvalue { i8, i8, i8, i8 } %a, 0
  store volatile i8 %0, i8* inttoptr (i64 4 to i8*)
  ; CHECK-NEXT: sts     5, r23
  %1 = extractvalue { i8, i8, i8, i8 } %a, 1
  store volatile i8 %1, i8* inttoptr (i64 5 to i8*)
  ; CHECK-NEXT: sts     6, r24
  %2 = extractvalue { i8, i8, i8, i8 } %a, 2
  store volatile i8 %2, i8* inttoptr (i64 6 to i8*)
  ; CHECK-NEXT: sts     7, r25
  %3 = extractvalue { i8, i8, i8, i8 } %a, 3
  store volatile i8 %3, i8* inttoptr (i64 7 to i8*)
  ret void
}

; CHECK-LABEL: ret_void_args_struct_i32_16_i8
define void @ret_void_args_struct_i32_16_i8({ i32, i16, i8} %a) {
start:
  ; CHECK:      sts     7, r21
  ; CHECK-NEXT: sts     6, r20
  ; CHECK-NEXT: sts     5, r19
  ; CHECK-NEXT: sts     4, r18
  %0 = extractvalue { i32, i16, i8 } %a, 0
  store volatile i32 %0, i32* inttoptr (i64 4 to i32*)

  ; CHECK-NEXT: sts     5, r23
  ; CHECK-NEXT: sts     4, r22
  %1 = extractvalue { i32, i16, i8 } %a, 1
  store volatile i16 %1, i16* inttoptr (i64 4 to i16*)

  ; CHECK-NEXT: sts     4, r24
  %2 = extractvalue { i32, i16, i8 } %a, 2
  store volatile i8 %2, i8* inttoptr (i64 4 to i8*)
  ret void
}

; CHECK-LABEL: ret_void_args_struct_i8_i32_struct_i32_i8
define void @ret_void_args_struct_i8_i32_struct_i32_i8({ i8, i32 } %a, { i32, i8 } %b) {
start:
  ; CHECK:      sts     4, r20
  %0 = extractvalue { i8, i32 } %a, 0
  store volatile i8 %0, i8* inttoptr (i64 4 to i8*)

  ; CHECK-NEXT: sts     8, r24
  ; CHECK-NEXT: sts     7, r23
  ; CHECK-NEXT: sts     6, r22
  ; CHECK-NEXT: sts     5, r21
  %1 = extractvalue { i8, i32 } %a, 1
  store volatile i32 %1, i32* inttoptr (i64 5 to i32*)

  ; CHECK-NEXT:      sts     9, r17
  ; CHECK-NEXT:      sts     8, r16
  ; CHECK-NEXT:      sts     7, r15
  ; CHECK-NEXT:      sts     6, r14
  %2 = extractvalue { i32, i8 } %b, 0
  store volatile i32 %2, i32* inttoptr (i64 6 to i32*)

  ; CHECK-NEXT: sts     7, r18
  %3 = extractvalue { i32, i8 } %b, 1
  store volatile i8 %3, i8* inttoptr (i64 7 to i8*)
  ret void
}

