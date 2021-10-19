; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;    struct sk_buff {
;      int i;
;      struct net_device *dev;
;    };
;    #define _(x) (__builtin_preserve_access_index(x))
;    static int (*bpf_probe_read)(void *dst, int size, void *unsafe_ptr)
;        = (void *) 4;
;
;    int bpf_prog(struct sk_buff *ctx) {
;      struct net_device *dev = 0;
;      bpf_probe_read(&dev, sizeof(dev), _(&ctx->dev));
;      return dev != 0;
;    }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.sk_buff = type { i32, %struct.net_device* }
%struct.net_device = type opaque

; Function Attrs: nounwind
define dso_local i32 @bpf_prog(%struct.sk_buff*) local_unnamed_addr #0 !dbg !15 {
  %2 = alloca %struct.net_device*, align 8
  call void @llvm.dbg.value(metadata %struct.sk_buff* %0, metadata !26, metadata !DIExpression()), !dbg !28
  %3 = bitcast %struct.net_device** %2 to i8*, !dbg !29
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #4, !dbg !29
  call void @llvm.dbg.value(metadata %struct.net_device* null, metadata !27, metadata !DIExpression()), !dbg !28
  store %struct.net_device* null, %struct.net_device** %2, align 8, !dbg !30, !tbaa !31
  %4 = tail call %struct.net_device** @llvm.preserve.struct.access.index.p0p0s_struct.net_devices.p0s_struct.sk_buffs(%struct.sk_buff* elementtype(%struct.sk_buff) %0, i32 1, i32 1), !dbg !35, !llvm.preserve.access.index !19
  %5 = bitcast %struct.net_device** %4 to i8*, !dbg !35
  %6 = call i32 inttoptr (i64 4 to i32 (i8*, i32, i8*)*)(i8* nonnull %3, i32 8, i8* %5) #4, !dbg !36
  %7 = load %struct.net_device*, %struct.net_device** %2, align 8, !dbg !37, !tbaa !31
  call void @llvm.dbg.value(metadata %struct.net_device* %7, metadata !27, metadata !DIExpression()), !dbg !28
  %8 = icmp ne %struct.net_device* %7, null, !dbg !38
  %9 = zext i1 %8 to i32, !dbg !38
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #4, !dbg !39
  ret i32 %9, !dbg !40
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   120
; CHECK-NEXT:        .long   120
; CHECK-NEXT:        .long   90
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 2)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   64                      # 0x40
; CHECK-NEXT:        .long   15                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   19                      # BTF_KIND_FWD(id = 5)
; CHECK-NEXT:        .long   117440512               # 0x7000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 6)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   30
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   34                      # BTF_KIND_FUNC(id = 7)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "sk_buff"               # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   105                     # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "dev"                   # string offset=11
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=15
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "net_device"            # string offset=19
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "ctx"                   # string offset=30
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "bpf_prog"              # string offset=34
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=43
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/llvm/test.c" # string offset=49
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "0:1"                   # string offset=86
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   124
; CHECK-NEXT:        .long   144
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   8                       # FuncInfo

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   43                      # Field reloc section string offset=43
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   86
; CHECK-NEXT:        .long   0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare %struct.net_device** @llvm.preserve.struct.access.index.p0p0s_struct.net_devices.p0s_struct.sk_buffs(%struct.sk_buff*, i32 immarg, i32 immarg) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "bpf_probe_read", scope: !0, file: !1, line: 6, type: !6, isLocal: true, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !9, !10}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)"}
!15 = distinct !DISubprogram(name: "bpf_prog", scope: !1, file: !1, line: 9, type: !16, scopeLine: 9, flags: DIFlagPrototyped, isLocal: false, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !25)
!16 = !DISubroutineType(types: !17)
!17 = !{!9, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "sk_buff", file: !1, line: 1, size: 128, elements: !20)
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !19, file: !1, line: 2, baseType: !9, size: 32)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "dev", scope: !19, file: !1, line: 3, baseType: !23, size: 64, offset: 64)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "net_device", file: !1, line: 3, flags: DIFlagFwdDecl)
!25 = !{!26, !27}
!26 = !DILocalVariable(name: "ctx", arg: 1, scope: !15, file: !1, line: 9, type: !18)
!27 = !DILocalVariable(name: "dev", scope: !15, file: !1, line: 10, type: !23)
!28 = !DILocation(line: 0, scope: !15)
!29 = !DILocation(line: 10, column: 3, scope: !15)
!30 = !DILocation(line: 10, column: 22, scope: !15)
!31 = !{!32, !32, i64 0}
!32 = !{!"any pointer", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 11, column: 37, scope: !15)
!36 = !DILocation(line: 11, column: 3, scope: !15)
!37 = !DILocation(line: 12, column: 10, scope: !15)
!38 = !DILocation(line: 12, column: 14, scope: !15)
!39 = !DILocation(line: 13, column: 1, scope: !15)
!40 = !DILocation(line: 12, column: 3, scope: !15)
