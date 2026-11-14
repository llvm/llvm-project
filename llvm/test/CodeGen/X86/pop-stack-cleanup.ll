; RUN: llc < %s -mtriple=i686-windows | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=LINUX64

declare void @param1(i32 %a)
declare i32 @param2_ret(i32 %a, i32 %b)
declare i64 @param2_ret64(i32 %a, i32 %b)
declare void @param2(i32 %a, i32 %b)
declare void @param3(i32 %a, i32 %b, i32 %c)
declare void @param8(i64, i64, i64, i64, i64, i64, i64, i64)
declare i32 @param8_ret(i64, i64, i64, i64, i64, i64, i64, i64)


define void @test() minsize nounwind {
; CHECK-LABEL: test:
; CHECK: calll _param1
; CHECK-NEXT: popl %eax
; CHECK: calll _param2
; CHECK-NEXT: popl %eax
; CHECK-NEXT: popl %ecx
; CHECK: calll _param2_ret
; CHECK-NEXT: popl %ecx
; CHECK-NEXT: popl %edx
; CHECK-NEXT: pushl %eax
; CHECK: calll _param3
; CHECK-NEXT: addl $12, %esp
; CHECK: calll _param2_ret64
; CHECK-NEXT: popl %ecx
; CHECK-NEXT: popl %ecx
  call void @param1(i32 1)
  call void @param2(i32 1, i32 2)
  %ret = call i32 @param2_ret(i32 1, i32 2)
  call void @param3(i32 1, i32 2, i32 %ret)
  %ret64 = call i64 @param2_ret64(i32 1, i32 2)  
  ret void
}

define void @debug_value_after_call(i32 %a) minsize nounwind !dbg !5 {
; CHECK-LABEL: debug_value_after_call:
; CHECK: calll _param2
; CHECK-NOT: addl $8, %esp
; CHECK: popl %eax
; CHECK-NEXT: popl %ecx
; CHECK: calll _param2
  call void @param2(i32 %a, i32 %a), !dbg !9
    #dbg_value(i32 5, !8, !DIExpression(), !9)
  call void @param2(i32 5, i32 %a), !dbg !9
  ret void, !dbg !9
}

define void @negative(i32 %k) {
; CHECK-LABEL: negative:
; CHECK: calll _param1
; CHECK-NEXT: addl $4, %esp
; CHECK: calll _param2
; CHECK-NEXT: addl $8, %esp
; CHECK: calll _param3
; CHECK-NEXT: movl %ebp, %esp
  %v = alloca i32, i32 %k
  call void @param1(i32 1)
  call void @param2(i32 1, i32 2)
  call void @param3(i32 1, i32 2, i32 3)
  ret void
}

define void @spill(i32 inreg %a, i32 inreg %b, i32 inreg %c) minsize nounwind {
; CHECK-LABEL: spill:
; CHECK-DAG: movl %ecx,
; CHECK-DAG: movl %edx,
; CHECK: calll _param2_ret
; CHECK-NEXT: popl %ecx
; CHECK-NEXT: popl %edx
; CHECK-DAG: movl {{.*}}, %ecx
; CHECK-DAG: movl {{.*}}, %edx
; CHECK: calll _spill
  %i = call i32 @param2_ret(i32 1, i32 2)
  call void @spill(i32 inreg %a, i32 inreg %b, i32 inreg %c)
  ret void
}

define void @test_linux64(i32 %size) minsize nounwind {
; LINUX64-LABEL: test_linux64:
; LINUX64: pushq %rbp
; LINUX64: callq param8
; LINUX64-NEXT: popq %rax
; LINUX64-NEXT: popq %rcx

  %a = alloca i64, i32 %size, align 8
  call void @param8(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  ret void
}

define i32 @test_linux64_i32(i32 %size) minsize nounwind {
; LINUX64-LABEL: test_linux64_i32:
; LINUX64: callq param8_ret
; LINUX64-NOT: popq %rax
; LINUX64: retq
  %a = alloca i64, i32 %size, align 8
  %r = call i32 @param8_ret(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  ret i32 %r
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "debug_value_after_call", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocalVariable(name: "i", scope: !5, file: !1, line: 2, type: !10)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
