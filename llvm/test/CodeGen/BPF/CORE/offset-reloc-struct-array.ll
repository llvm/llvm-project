; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct net_device {
;     int dev_id;
;     int others;
;   };
;   struct sk_buff {
;     int i;
;     struct net_device dev[10];
;   };
;   #define _(x) (__builtin_preserve_access_index(x))
;   static int (*bpf_probe_read)(void *dst, int size, void *unsafe_ptr)
;       = (void *) 4;
;
;   int bpf_prog(struct sk_buff *ctx) {
;     int dev_id;
;     bpf_probe_read(&dev_id, sizeof(int), _(&ctx->dev[5].dev_id));
;     return dev_id;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.sk_buff = type { i32, [10 x %struct.net_device] }
%struct.net_device = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @bpf_prog(%struct.sk_buff*) local_unnamed_addr #0 !dbg !15 {
  %2 = alloca i32, align 4
  call void @llvm.dbg.value(metadata %struct.sk_buff* %0, metadata !31, metadata !DIExpression()), !dbg !33
  %3 = bitcast i32* %2 to i8*, !dbg !34
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3) #4, !dbg !34
  %4 = tail call [10 x %struct.net_device]* @llvm.preserve.struct.access.index.p0a10s_struct.net_devices.p0s_struct.sk_buffs(%struct.sk_buff* %0, i32 1, i32 1), !dbg !35, !llvm.preserve.access.index !19
  %5 = tail call %struct.net_device* @llvm.preserve.array.access.index.p0s_struct.net_devices.p0a10s_struct.net_devices([10 x %struct.net_device]* %4, i32 1, i32 5), !dbg !35, !llvm.preserve.access.index !23
  %6 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.net_devices(%struct.net_device* %5, i32 0, i32 0), !dbg !35, !llvm.preserve.access.index !24
  %7 = bitcast i32* %6 to i8*, !dbg !35
  %8 = call i32 inttoptr (i64 4 to i32 (i8*, i32, i8*)*)(i8* nonnull %3, i32 4, i8* %7) #4, !dbg !36
  %9 = load i32, i32* %2, align 4, !dbg !37, !tbaa !38
  call void @llvm.dbg.value(metadata i32 %9, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3) #4, !dbg !42
  ret i32 %9, !dbg !43
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   172
; CHECK-NEXT:        .long   172
; CHECK-NEXT:        .long   128
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 2)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   84
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   15                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   19                      # BTF_KIND_STRUCT(id = 4)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   30
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   37
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   0                       # BTF_KIND_ARRAY(id = 5)
; CHECK-NEXT:        .long   50331648                # 0x3000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   44                      # BTF_KIND_INT(id = 6)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 7)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   68                      # BTF_KIND_FUNC(id = 8)
; CHECK-NEXT:        .long   201326592               # 0xc000000
; CHECK-NEXT:        .long   7
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
; CHECK-NEXT:        .ascii  "dev_id"                # string offset=30
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "others"                # string offset=37
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "__ARRAY_SIZE_TYPE__"   # string offset=44
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "ctx"                   # string offset=64
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "bpf_prog"              # string offset=68
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=77
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/llvm/test.c" # string offset=83
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "0:1:5:0"               # string offset=120
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   8                       # FuncInfo

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   77                      # Field reloc section string offset=77
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   120
; CHECK-NEXT:        .long   0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare [10 x %struct.net_device]* @llvm.preserve.struct.access.index.p0a10s_struct.net_devices.p0s_struct.sk_buffs(%struct.sk_buff*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare %struct.net_device* @llvm.preserve.array.access.index.p0s_struct.net_devices.p0a10s_struct.net_devices([10 x %struct.net_device]*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.net_devices(%struct.net_device*, i32 immarg, i32 immarg) #2

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
!5 = distinct !DIGlobalVariable(name: "bpf_probe_read", scope: !0, file: !1, line: 10, type: !6, isLocal: true, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !9, !10}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)"}
!15 = distinct !DISubprogram(name: "bpf_prog", scope: !1, file: !1, line: 13, type: !16, scopeLine: 13, flags: DIFlagPrototyped, isLocal: false, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !30)
!16 = !DISubroutineType(types: !17)
!17 = !{!9, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "sk_buff", file: !1, line: 5, size: 672, elements: !20)
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !19, file: !1, line: 6, baseType: !9, size: 32)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "dev", scope: !19, file: !1, line: 7, baseType: !23, size: 640, offset: 32)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 640, elements: !28)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "net_device", file: !1, line: 1, size: 64, elements: !25)
!25 = !{!26, !27}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "dev_id", scope: !24, file: !1, line: 2, baseType: !9, size: 32)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "others", scope: !24, file: !1, line: 3, baseType: !9, size: 32, offset: 32)
!28 = !{!29}
!29 = !DISubrange(count: 10)
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "ctx", arg: 1, scope: !15, file: !1, line: 13, type: !18)
!32 = !DILocalVariable(name: "dev_id", scope: !15, file: !1, line: 14, type: !9)
!33 = !DILocation(line: 0, scope: !15)
!34 = !DILocation(line: 14, column: 3, scope: !15)
!35 = !DILocation(line: 15, column: 40, scope: !15)
!36 = !DILocation(line: 15, column: 3, scope: !15)
!37 = !DILocation(line: 16, column: 10, scope: !15)
!38 = !{!39, !39, i64 0}
!39 = !{!"int", !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C/C++ TBAA"}
!42 = !DILocation(line: 17, column: 1, scope: !15)
!43 = !DILocation(line: 16, column: 3, scope: !15)
