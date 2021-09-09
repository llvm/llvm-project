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
  call void @sink(i8* %p)
  ret void
}


; NOPERS: personality void ()* @global
; PERS: personality {{.*}} @__hwasan_personality_thunk.global
define void @stack2() sanitize_hwaddress personality void ()* @global {
  %p = alloca i8
  call void @sink(i8* %p)
  ret void
}

define internal void @local() {
  ret void
}

@local_alias = internal alias void (), void ()* @local

; NOPERS: personality void ()* @local
; PERS: personality {{.*}} @__hwasan_personality_thunk.local
define void @stack3() sanitize_hwaddress personality void ()* @local {
  %p = alloca i8
  call void @sink(i8* %p)
  ret void
}

; NOPERS: personality void ()* @local_alias
; PERS: personality {{.*}} @__hwasan_personality_thunk.local_alias
define void @stack4() sanitize_hwaddress personality void ()* @local_alias {
  %p = alloca i8
  call void @sink(i8* %p)
  ret void
}

; NOPERS: personality void ()* inttoptr (i64 1 to void ()*)
; PERS: personality i32 (i32, i32, i64, i8*, i8*)* @__hwasan_personality_thunk.
define void @stack5() sanitize_hwaddress personality void ()* inttoptr (i64 1 to void ()*) {
  %p = alloca i8
  call void @sink(i8* %p)
  ret void
}

; NOPERS: personality void ()* inttoptr (i64 2 to void ()*)
; PERS: personality i32 (i32, i32, i64, i8*, i8*)* @__hwasan_personality_thunk..1
define void @stack6() sanitize_hwaddress personality void ()* inttoptr (i64 2 to void ()*) {
  %p = alloca i8
  call void @sink(i8* %p)
  ret void
}

declare void @global()
declare void @sink(i8*)

; PERS: define linkonce_odr hidden i32 @__hwasan_personality_thunk(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4) comdat
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4, i8* null, i8* bitcast (void ()* @_Unwind_GetGR to i8*), i8* bitcast (void ()* @_Unwind_GetCFA to i8*))
; PERS: ret i32 %5

; PERS: define linkonce_odr hidden i32 @__hwasan_personality_thunk.global(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4) comdat
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4, i8* bitcast (void ()* @global to i8*), i8* bitcast (void ()* @_Unwind_GetGR to i8*), i8* bitcast (void ()* @_Unwind_GetCFA to i8*))
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk.local(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4)
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4, i8* bitcast (void ()* @local to i8*), i8* bitcast (void ()* @_Unwind_GetGR to i8*), i8* bitcast (void ()* @_Unwind_GetCFA to i8*))
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk.local_alias(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4)
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4, i8* bitcast (void ()* @local_alias to i8*), i8* bitcast (void ()* @_Unwind_GetGR to i8*), i8* bitcast (void ()* @_Unwind_GetCFA to i8*))
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk.(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4) {
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4, i8* inttoptr (i64 1 to i8*), i8* bitcast (void ()* @_Unwind_GetGR to i8*), i8* bitcast (void ()* @_Unwind_GetCFA to i8*))
; PERS: ret i32 %5

; PERS: define internal i32 @__hwasan_personality_thunk..1(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4) {
; PERS: %5 = tail call i32 @__hwasan_personality_wrapper(i32 %0, i32 %1, i64 %2, i8* %3, i8* %4, i8* inttoptr (i64 2 to i8*), i8* bitcast (void ()* @_Unwind_GetGR to i8*), i8* bitcast (void ()* @_Unwind_GetCFA to i8*))
; PERS: ret i32 %5
