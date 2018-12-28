;__kernel void foo(){}
;int bar(int a) {
;    a += 3;
;    int b = 10;
;    while (b) {
;        if (a > 4)
;            b = a - 1;
;        else  {
;            b = a + 1;
;            a = a *3;
;        }
;    }
;    return b;
;}
; RUN: llvm-as < %s | llvm-spirv -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-as < %s | llvm-spirv -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o -| FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; XFAIL: *
; This is requires debug metadata update.
; We also should remove checks for LLVM IR stuff like labels, names, etc.

; CHECK-SPIRV: String [[str:[0-9]+]] "/tmp.cl"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; CHECK-SPIRV: 4 Line [[str]] 1 0
; CHECK-SPIRV-NEXT: Function
; Function Attrs: nounwind
define spir_kernel void @foo() #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 {
entry:
; CHECK-LLVM: ret void, !dbg ![[Line_1:[0-9]+]]
ret void, !dbg !23
}
; CHECK-SPIRV: FunctionEnd

; CHECK-SPIRV: 4 Line [[str]] 2 0
; CHECK-SPIRV-NEXT: Function
; Function Attrs: nounwind
define spir_func i32 @bar(i32 %a) #0 {
; CHECK-SPIRV: Label [[entry:[0-9]+]]
entry:
  call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !24, metadata !25), !dbg !26
; CHECK-SPIRV: 4 Line [[str]] 3 0
; CHECK-SPIRV-NEXT: IAdd
; CHECK-LLVM: %add = add i32 %a, 3, !dbg ![[Line_3:[0-9]+]]
  %add = add nsw i32 %a, 3, !dbg !27
  call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !24, metadata !25), !dbg !26
  call void @llvm.dbg.value(metadata i32 10, i64 0, metadata !28, metadata !25), !dbg !29
; CHECK-SPIRV: 4 Line [[str]] 5 0
; CHECK-SPIRV-NEXT: Branch [[while_cond:[0-9]+]]
; CHECK-LLVM: br label %while.cond, !dbg ![[Line_5:[0-9]+]]
  br label %while.cond, !dbg !30

; CHECK-SPIRV: Label [[while_cond]]
while.cond:                                       ; preds = %if.end, %entry
  %b.0 = phi i32 [ 10, %entry ], [ %b.1, %if.end ]
  %a.addr.0 = phi i32 [ %add, %entry ], [ %a.addr.1, %if.end ]
; CHECK-SPIRV: 4 Line [[str]] 5 0
; CHECK-SPIRV-NEXT: INotEqual
; CHECK-LLVM: %tobool = icmp ne i32 %b.0, 0, !dbg ![[Line_5]]
  %tobool = icmp ne i32 %b.0, 0, !dbg !31
; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} [[while_body:[0-9]+]] [[while_end:[0-9]+]]
; CHECK-LLVM: br i1 %tobool, label %while.body, label %while.end, !dbg ![[Line_5]]
  br i1 %tobool, label %while.body, label %while.end, !dbg !31

; CHECK-SPIRV: Label [[while_body]]
while.body:                                       ; preds = %while.cond
; CHECK-SPIRV: 4 Line [[str]] 6 0
; CHECK-SPIRV-NEXT: SGreaterThan
; CHECK-LLVM: %cmp = icmp sgt i32 %a.addr.0, 4, !dbg ![[Line_6:[0-9]+]]
  %cmp = icmp sgt i32 %a.addr.0, 4, !dbg !34
; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} [[if_then:[0-9]+]] [[if_else:[0-9]+]]
; CHECK-LLVM: br i1 %cmp, label %if.then, label %if.else, !dbg ![[Line_6]]
  br i1 %cmp, label %if.then, label %if.else, !dbg !37

; CHECK-SPIRV: Label [[if_then]]
if.then:                                          ; preds = %while.body
; CHECK-SPIRV: 4 Line [[str]] 7 0
; CHECK-SPIRV-NEXT: ISub
; CHECK-LLVM: %sub = sub i32 %a.addr.0, 1, !dbg ![[Line_7:[0-9]+]]
  %sub = sub nsw i32 %a.addr.0, 1, !dbg !38
  call void @llvm.dbg.value(metadata i32 %sub, i64 0, metadata !28, metadata !25), !dbg !29
; CHECK-SPIRV-NEXT: Branch [[if_end:[0-9]+]]
; CHECK-LLVM: br label %if.end, !dbg ![[Line_7]]
  br label %if.end, !dbg !38

; CHECK-SPIRV: Label [[if_else]]
if.else:                                          ; preds = %while.body
; CHECK-SPIRV: 4 Line [[str]] 9 0
; CHECK-SPIRV-NEXT: IAdd
; CHECK-LLVM: %add1 = add i32 %a.addr.0, 1, !dbg ![[Line_9:[0-9]+]]
  %add1 = add nsw i32 %a.addr.0, 1, !dbg !39
  call void @llvm.dbg.value(metadata i32 %add1, i64 0, metadata !28, metadata !25), !dbg !29
; CHECK-SPIRV: 4 Line [[str]] 10 0
; CHECK-SPIRV-NEXT: IMul
; CHECK-LLVM: %mul = mul  i32 %a.addr.0, 3, !dbg ![[Line_10:[0-9]+]]
  %mul = mul nsw i32 %a.addr.0, 3, !dbg !41
  call void @llvm.dbg.value(metadata i32 %mul, i64 0, metadata !24, metadata !25), !dbg !26
; CHECK-SPIRV-NEXT: Branch [[if_end]]
; CHECK-LLVM: br label %if.end, !dbg ![[Line_10]]
  br label %if.end

; CHECK-SPIRV: Label [[if_end]]
if.end:                                           ; preds = %if.else, %if.then
  %b.1 = phi i32 [ %sub, %if.then ], [ %add1, %if.else ]
  %a.addr.1 = phi i32 [ %a.addr.0, %if.then ], [ %mul, %if.else ]
; CHECK-SPIRV: 4 Line [[str]] 5 0
; CHECK-SPIRV-NEXT: Branch [[while_cond]]
; CHECK-LLVM: br label %while.cond, !dbg ![[Line_5]]
  br label %while.cond, !dbg !30

; CHECK-SPIRV: Label [[while_end]]
while.end:                                        ; preds = %while.cond
; CHECK-SPIRV: 4 Line [[str]] 13 0
; CHECK-SPIRV-NEXT: ReturnValue
; CHECK-LLVM: ret i32 %b.0, !dbg ![[Line_13:[0-9]+]]
  ret i32 %b.0, !dbg !42
}
; CHECK-SPIRV: FunctionEnd

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !20}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!21}
!opencl.ocl.version = !{!21}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!22}

!0 = !{!"0x11\0012\00clang version 3.6.1 (https://github.com/KhronosGroup/SPIR.git 5df927fb80a9b807bf12eb82ae686c3978b4ccc7) (https://github.com/KhronosGroup/SPIRV-LLVM.git dcefbf9bcd9eb48c41aa16935e5a4e17d8160c8a)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp//<stdin>] [DW_LANG_C99]
!1 = !{!"/<stdin>", !"/spirv_limits_characters_in_a_literal_name"}
!2 = !{}
!3 = !{!4, !9}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\000\000\001", !5, !6, !7, null, void ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"/tmp.cl", !"/spirv_limits_characters_in_a_literal_name"}
!6 = !{!"0x29", !5}                               ; [ DW_TAG_file_type ] [/tmp//tmp.cl]
!7 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!9 = !{!"0x2e\00bar\00bar\00\002\000\001\000\000\00256\000\002", !5, !6, !10, null, i32 (i32)* @bar, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [bar]
!10 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{!12, !12}
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 2}
!21 = !{i32 1, i32 2}
!22 = !{!"clang version 3.6.1 (https://github.com/KhronosGroup/SPIR.git 5df927fb80a9b807bf12eb82ae686c3978b4ccc7) (https://github.com/KhronosGroup/SPIRV-LLVM.git dcefbf9bcd9eb48c41aa16935e5a4e17d8160c8a)"}
!23 = !MDLocation(line: 1, scope: !4)
!24 = !{!"0x101\00a\0016777218\000", !9, !6, !12} ; [ DW_TAG_arg_variable ] [a] [line 2]
!25 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!26 = !MDLocation(line: 2, scope: !9)
!27 = !MDLocation(line: 3, scope: !9)
!28 = !{!"0x100\00b\004\000", !9, !6, !12}        ; [ DW_TAG_auto_variable ] [b] [line 4]
!29 = !MDLocation(line: 4, scope: !9)
!30 = !MDLocation(line: 5, scope: !9)
!31 = !MDLocation(line: 5, scope: !32)
!32 = !{!"0xb\002", !5, !33}                      ; [ DW_TAG_lexical_block ] [/tmp//tmp.cl]
!33 = !{!"0xb\001", !5, !9}                       ; [ DW_TAG_lexical_block ] [/tmp//tmp.cl]
!34 = !MDLocation(line: 6, scope: !35)
!35 = !{!"0xb\006\000\001", !5, !36}              ; [ DW_TAG_lexical_block ] [/tmp//tmp.cl]
!36 = !{!"0xb\005\000\000", !5, !9}               ; [ DW_TAG_lexical_block ] [/tmp//tmp.cl]
!37 = !MDLocation(line: 6, scope: !36)
!38 = !MDLocation(line: 7, scope: !35)
!39 = !MDLocation(line: 9, scope: !40)
!40 = !{!"0xb\008\000\002", !5, !35}              ; [ DW_TAG_lexical_block ] [/tmp//tmp.cl]
!41 = !MDLocation(line: 10, scope: !40)
!42 = !MDLocation(line: 13, scope: !9)

; CHECK-LLVM: ![[Line_1]] = !MDLocation(line: 1
; CHECK-LLVM: ![[Line_3]] = !MDLocation(line: 3
; CHECK-LLVM: ![[Line_5]] = !MDLocation(line: 5
; CHECK-LLVM: ![[Line_6]] = !MDLocation(line: 6
; CHECK-LLVM: ![[Line_7]] = !MDLocation(line: 7
; CHECK-LLVM: ![[Line_9]] = !MDLocation(line: 9
; CHECK-LLVM: ![[Line_10]] = !MDLocation(line: 10
; CHECK-LLVM: ![[Line_13]] = !MDLocation(line: 13

