; RUN: llc -verify-machineinstrs -mtriple=wasm32-unknown-unknown < %s | FileCheck -check-prefix=WASM32 %s

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"		; <ptr> [#uses=1]

; WASM32-LABEL: test:
; WASM32:      i32.load        28
; WASM32:      br_if           0
; WASM32:      call __stack_chk_fail
; WASM32-NEXT: unreachable

define void @test(ptr %a) nounwind ssp {
entry:
	%a_addr = alloca ptr		; <ptr> [#uses=2]
	%buf = alloca [8 x i8]		; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %a, ptr %a_addr
	%0 = load ptr, ptr %a_addr, align 4		; <ptr> [#uses=1]
	%1 = call ptr @strcpy(ptr %buf, ptr %0) nounwind		; <ptr> [#uses=0]
	%2 = call i32 (ptr, ...) @printf(ptr @"\01LC", ptr %buf) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

; WASM32-LABEL: test_return_i32:
; WASM32:      call __stack_chk_fail
; WASM32-NEXT: unreachable

define i32 @test_return_i32(ptr %a) nounwind ssp {
entry:
  %a_addr = alloca ptr    ; <ptr> [#uses=2]
  %buf = alloca [8 x i8]    ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32    ; <i32> [#uses=0]
  store ptr %a, ptr %a_addr
  %0 = load ptr, ptr %a_addr, align 4    ; <ptr> [#uses=1]
  %1 = call ptr @strcpy(ptr %buf, ptr %0) nounwind    ; <ptr> [#uses=0]
  %2 = call i32 (ptr, ...) @printf(ptr @"\01LC", ptr %buf) nounwind    ; <i32> [#uses=0]
  br label %return

return:    ; preds = %entry
  ret i32 0
}

declare ptr @strcpy(ptr, ptr) nounwind

declare i32 @printf(ptr, ...) nounwind
