import sys

# Put 32768 one-byte DW_OP_dup operations between the branch and its label,
# which gives us the first offset that won't fit.
DUPLICATE_OPS = "DW_OP_dup, " * 32768

sys.stdout.write(
    f"""\
define void @f() !dbg !5 {{
entry:
  #dbg_value(i64 0, !9,
             !DIExpression(DW_OP_LLVM_bra, 1, {DUPLICATE_OPS}
                           DW_OP_LLVM_label, 1), !10)
  ret void, !dbg !10
}}

!llvm.dbg.cu = !{{!0}}
!llvm.module.flags = !{{!3}}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{{i32 2, !"Debug Info Version", i32 3}}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !6,
                            spFlags: DISPFlagDefinition,
                            unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{{null}}
!9 = !DILocalVariable(name: "x", scope: !5, type: !11)
!10 = !DILocation(line: 1, scope: !5)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
"""
)
