; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare i31 @"printf"(i8*, ...)   ;; Prototype for: i32 __builtin_printf(const char*, ...)

define i31 @"testvarar"()
begin
	call i31(i8*, ...) *@printf(i8 * null, i31 12, i8 42);
	ret i31 %1
end


