; RUN: opt < %s -mtriple aarch64-linux-android29 -passes=hwasan -S | FileCheck %s --check-prefix=NOPERS
; RUN: opt < %s -mtriple aarch64-linux-android30 -passes=hwasan -S | FileCheck %s --check-prefix=PERS

; NOPERS: define void @nostack() #{{[0-9]+}} {
; PERS: define void @nostack() #{{[0-9]+}} {
define void @nostack() sanitize_hwaddress {
  ret void
}

; NOPERS: define void @stack1() #{{[0-9]+}} {
; PERS: personality {{.*}} @__hwasan_personality_thunk
define void @stack1() sanitize_hwaddress {
  %p = alloca i8
  call void @sink(ptr %p)
  ret void
}


; NOPERS: personality ptr @global
; PERS: personality {{.*}} @__hwasan_personality_thunk.global
define void @stack2() sanitize_hwaddress personality ptr @global {
  %p = alloca i8
  call void @sink(ptr %p)
  ret void
}

define internal void @local() {
  ret void
}

@local_alias = internal alias void (), ptr @local

; NOPERS: personality ptr @local
; PERS: personality {{.*}} @__hwasan_personality_thunk.local
define void @stack3() sanitize_hwaddress personality ptr @local {
  %p = alloca i8
  call void @sink(ptr %p)
  ret void
}

; NOPERS: personality ptr @local_alias
; PERS: personality {{.*}} @__hwasan_personality_thunk.local_alias
define void @stack4() sanitize_hwaddress personality ptr @local_alias {
  %p = alloca i8
  call void @sink(ptr %p)
  ret void
}

; NOPERS: personality ptr inttoptr (i64 1 to ptr)
; PERS: personality ptr @__hwasan_personality_thunk.
define void @stack5() sanitize_hwaddress personality ptr inttoptr (i64 1 to ptr) {
  %p = alloca i8
  call void @sink(ptr %p)
  ret void
}

; NOPERS: personality ptr inttoptr (i64 2 to ptr)
; PERS: personality ptr @__hwasan_personality_thunk..1
define void @stack6() sanitize_hwaddress personality ptr inttoptr (i64 2 to ptr) {
  %p = alloca i8
  call void @sink(ptr %p)
  ret void
}

declare void @global()
declare void @sink(ptr)

; PERS: define linkonce_odr hidden i32 @__hwasan_personality_thunk(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4) comdat
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4, ptr null, ptr @_Unwind_GetGR, ptr @_Unwind_GetCFA)
; PERS: ret i32 %5

; PERS: define linkonce_odr hidden i32 @__hwasan_personality_thunk.global(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4) comdat
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4, ptr @global, ptr @_Unwind_GetGR, ptr @_Unwind_GetCFA)
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk.local(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4)
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4, ptr @local, ptr @_Unwind_GetGR, ptr @_Unwind_GetCFA)
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk.local_alias(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4)
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4, ptr @local_alias, ptr @_Unwind_GetGR, ptr @_Unwind_GetCFA)
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk.(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4) {
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4, ptr inttoptr (i64 1 to ptr), ptr @_Unwind_GetGR, ptr @_Unwind_GetCFA)
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk..1(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4) {
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, ptr %3, ptr %4, ptr inttoptr (i64 2 to ptr), ptr @_Unwind_GetGR, ptr @_Unwind_GetCFA)
; PERS: ret i32 %5
