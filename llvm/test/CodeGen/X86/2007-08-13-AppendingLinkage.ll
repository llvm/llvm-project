; RUN: llc < %s -march=x86 | not grep drectve
; PR1607

%hlvm_programs_element = type { i8*, i32 (i32, i8**)* }
@hlvm_programs = appending constant [1 x %hlvm_programs_element]
zeroinitializer

define %hlvm_programs_element* @hlvm_get_programs() {
entry:
  ret %hlvm_programs_element* getelementptr([1 x %hlvm_programs_element]*  
                                            @hlvm_programs, i32 0, i32 0)
}
