; RUN: opt -S -dxil-resource-type -dxil-resource-access -mtriple=dxil -verify-each %s | FileCheck %s
; RUN: opt -S -passes='require<dxil-resource-type>,function(require<domtree>,dxil-resource-access,verify<domtree>)' -mtriple=dxil -verify-each %s | FileCheck %s
; RUN: opt -disable-output -passes='require<dxil-resource-type>,function(dxil-resource-access),scalarizer,dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library -verify-each %s

%CB = type { [3 x <4 x i32>] }

; Look through arithmetic on each PHI edge to prove the byte offset is aligned.
; CHECK-LABEL: define <3 x i32> @row_aligned_phi(
; CHECK: merge:
; CHECK-NEXT: [[OFFSET:%.*]] = phi i32
; CHECK-NEXT: [[ROW:%.*]] = udiv exact i32 [[OFFSET]], 16
; CHECK-NEXT: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %CB) {{%.*}}, i32 [[ROW]])
; CHECK-NEXT: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK-NEXT: [[Y:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 1
; CHECK-NEXT: [[Z:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 2
; CHECK-NEXT: [[V0:%.*]] = insertelement <3 x i32> poison, i32 [[X]], i32 0
; CHECK-NEXT: [[V1:%.*]] = insertelement <3 x i32> [[V0]], i32 [[Y]], i32 1
; CHECK-NEXT: [[V2:%.*]] = insertelement <3 x i32> [[V1]], i32 [[Z]], i32 2
; CHECK-NEXT: ret <3 x i32> [[V2]]
define <3 x i32> @row_aligned_phi(i1 %cond, i32 %row) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  br i1 %cond, label %then, label %merge

then:
  %scaled = mul i32 %row, 16
  %offset_input = add i32 0, %scaled
  br label %merge

merge:
  %offset = phi i32 [ 0, %entry ], [ %offset_input, %then ]
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <3 x i32>, ptr addrspace(2) %ptr, align 16
  ret <3 x i32> %value
}

; A nonzero component on one edge must retain the conditional second-row load.
; CHECK-LABEL: define <2 x i32> @partially_aligned_phi(
; CHECK: merge:
; CHECK: [[ROW:%.*]] = udiv i32 {{%.*}}, 16
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 4
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK: [[LAST:%.*]] = add i32 [[COMP]], 1
; CHECK-NEXT: [[CROSS:%.*]] = icmp uge i32 [[LAST]], 4
; CHECK-NEXT: br i1 [[CROSS]], label %[[NEXT:.*]], label %[[MERGE:.*]]
; CHECK: [[NEXT]]:
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK: [[MERGE]]:
; CHECK-NEXT: [[WINDOW:%.*]] = phi <8 x i32>
; CHECK-COUNT-2: extractelement <8 x i32> [[WINDOW]], i32
; CHECK: ret <2 x i32>
define <2 x i32> @partially_aligned_phi(i1 %cond, i32 %row) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  br i1 %cond, label %then, label %merge

then:
  %scaled = mul i32 %row, 16
  %offset_input = add i32 %scaled, 12
  br label %merge

merge:
  %offset = phi i32 [ 0, %entry ], [ %offset_input, %then ]
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <2 x i32>, ptr addrspace(2) %ptr, align 4
  ret <2 x i32> %value
}

; Dynamic components 0 and 1 both leave room for three elements in one row.
; CHECK-LABEL: define <3 x i32> @single_row(
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 4
; CHECK-NOT: br
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK: [[WINDOW:%.*]] = insertelement <4 x i32> {{%.*}}, i32 {{%.*}}, i32 3
; CHECK-NEXT: [[X:%.*]] = extractelement <4 x i32> [[WINDOW]], i32 [[COMP]]
; CHECK-NEXT: [[V0:%.*]] = insertelement <3 x i32> poison, i32 [[X]], i32 0
; CHECK-NEXT: [[I1:%.*]] = add i32 [[COMP]], 1
; CHECK-NEXT: [[Y:%.*]] = extractelement <4 x i32> [[WINDOW]], i32 [[I1]]
; CHECK-NEXT: [[V1:%.*]] = insertelement <3 x i32> [[V0]], i32 [[Y]], i32 1
; CHECK-NEXT: [[I2:%.*]] = add i32 [[COMP]], 2
; CHECK-NEXT: [[Z:%.*]] = extractelement <4 x i32> [[WINDOW]], i32 [[I2]]
; CHECK-NEXT: [[V2:%.*]] = insertelement <3 x i32> [[V1]], i32 [[Z]], i32 2
; CHECK-NEXT: ret <3 x i32> [[V2]]
define <3 x i32> @single_row(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 4
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <3 x i32>, ptr addrspace(2) %ptr, align 4
  ret <3 x i32> %value
}

; Do not access the second row when the pair stays in the first row.
; CHECK-LABEL: define <2 x i32> @conditional_crossing(
; CHECK: [[QUOTIENT:%.*]] = udiv i32 {{%.*}}, 16
; CHECK-NEXT: [[ROW:%.*]] = add i32 0, [[QUOTIENT]]
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 4
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %CB) {{%.*}}, i32 [[ROW]])
; CHECK-NOT: call
; CHECK: [[FIRST:%.*]] = insertelement <8 x i32> {{%.*}}, i32 {{%.*}}, i32 3
; CHECK-NEXT: [[LAST:%.*]] = add i32 [[COMP]], 1
; CHECK-NEXT: [[CROSS:%.*]] = icmp uge i32 [[LAST]], 4
; CHECK-NEXT: br i1 [[CROSS]], label %[[NEXT:.*]], label %[[MERGE:.*]]
; CHECK: [[NEXT]]:
; CHECK-NEXT: [[ROW1:%.*]] = add i32 [[ROW]], 1
; CHECK-NEXT: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %CB) {{%.*}}, i32 [[ROW1]])
; CHECK-NOT: call
; CHECK: [[SECOND:%.*]] = insertelement <8 x i32> {{%.*}}, i32 {{%.*}}, i32 7
; CHECK-NEXT: br label %[[MERGE]]
; CHECK: [[MERGE]]:
; CHECK-NEXT: [[WINDOW:%.*]] = phi <8 x i32> [ [[FIRST]], %entry ], [ [[SECOND]], %[[NEXT]] ]
; CHECK-NEXT: [[X:%.*]] = extractelement <8 x i32> [[WINDOW]], i32 [[COMP]]
; CHECK-NEXT: [[V0:%.*]] = insertelement <2 x i32> poison, i32 [[X]], i32 0
; CHECK-NEXT: [[I1:%.*]] = add i32 [[COMP]], 1
; CHECK-NEXT: [[Y:%.*]] = extractelement <8 x i32> [[WINDOW]], i32 [[I1]]
; CHECK-NEXT: [[V1:%.*]] = insertelement <2 x i32> [[V0]], i32 [[Y]], i32 1
; CHECK-NEXT: ret <2 x i32> [[V1]]
define <2 x i32> @conditional_crossing(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 12
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <2 x i32>, ptr addrspace(2) %ptr, align 4
  ret <2 x i32> %value
}

; In a profiled function the row-crossing branch gets explicitly unknown
; branch weights.
; CHECK-LABEL: define <2 x i32> @conditional_crossing_profiled(
; CHECK: br i1 {{%.*}}, label {{%.*}}, label {{%.*}}, !prof [[UNKNOWN:![0-9]+]]
define <2 x i32> @conditional_crossing_profiled(i1 %cond) !prof !0 {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 12
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <2 x i32>, ptr addrspace(2) %ptr, align 4
  ret <2 x i32> %value
}

; Five elements require exactly two rows for every starting component.
; CHECK-LABEL: define <5 x i32> @two_rows(
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK-NOT: br
; CHECK: [[WINDOW:%.*]] = insertelement <8 x i32> {{%.*}}, i32 {{%.*}}, i32 7
; CHECK-COUNT-5: extractelement <8 x i32> [[WINDOW]], i32
; CHECK-NOT: call
; CHECK-NOT: br
; CHECK: ret <5 x i32>
define <5 x i32> @two_rows(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 12
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <5 x i32>, ptr addrspace(2) %ptr, align 4
  ret <5 x i32> %value
}

; Known nonzero components can prove a short vector always needs two rows.
; CHECK-LABEL: define <3 x i32> @always_crossing(
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK-NOT: br
; CHECK: [[WINDOW:%.*]] = insertelement <8 x i32> {{%.*}}, i32 {{%.*}}, i32 7
; CHECK-COUNT-3: extractelement <8 x i32> [[WINDOW]], i32
; CHECK-NOT: call
; CHECK-NOT: br
; CHECK: ret <3 x i32>
define <3 x i32> @always_crossing(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 8, i32 12
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <3 x i32>, ptr addrspace(2) %ptr, align 4
  ret <3 x i32> %value
}

; Two rows are always needed; a third is needed only for nonzero components.
; CHECK-LABEL: define <8 x i32> @conditional_third_row(
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 4
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK: [[FIRST:%.*]] = insertelement <12 x i32> {{%.*}}, i32 {{%.*}}, i32 7
; CHECK-NEXT: [[LAST:%.*]] = add i32 [[COMP]], 7
; CHECK-NEXT: [[CROSS:%.*]] = icmp uge i32 [[LAST]], 8
; CHECK-NEXT: br i1 [[CROSS]], label %[[NEXT:.*]], label %[[MERGE:.*]]
; CHECK: [[NEXT]]:
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}
; CHECK-NOT: call
; CHECK: [[SECOND:%.*]] = insertelement <12 x i32> {{%.*}}, i32 {{%.*}}, i32 11
; CHECK-NEXT: br label %[[MERGE]]
; CHECK: [[MERGE]]:
; CHECK-NEXT: [[WINDOW:%.*]] = phi <12 x i32> [ [[FIRST]], %entry ], [ [[SECOND]], %[[NEXT]] ]
; CHECK-COUNT-8: extractelement <12 x i32> [[WINDOW]], i32
; CHECK-NOT: call
; CHECK: ret <8 x i32>
define <8 x i32> @conditional_third_row(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 12
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <8 x i32>, ptr addrspace(2) %ptr, align 4
  ret <8 x i32> %value
}

; A nonzero base remainder participates in both component selection and carry.
; CHECK-LABEL: define i32 @base_remainder(
; CHECK: [[TAIL:%.*]] = add i32 {{%.*}}, 12
; CHECK-NEXT: [[ROW:%.*]] = udiv i32 [[TAIL]], 16
; CHECK-NEXT: [[FIRST:%.*]] = add i32 1, [[ROW]]
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 4
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %CB) {{%.*}}, i32 [[FIRST]])
; CHECK-NOT: call
; CHECK-NOT: br
; CHECK: [[WINDOW:%.*]] = insertelement <4 x i32> {{%.*}}, i32 {{%.*}}, i32 3
; CHECK-NEXT: [[VALUE:%.*]] = extractelement <4 x i32> [[WINDOW]], i32 [[COMP]]
; CHECK-NEXT: ret i32 [[VALUE]]
define i32 @base_remainder(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 28)
  %offset = select i1 %cond, i32 0, i32 4
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load i32, ptr addrspace(2) %ptr, align 4
  ret i32 %value
}

; CHECK-LABEL: define <2 x i64> @double_width(
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 8
; CHECK: call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}
; CHECK: [[LAST:%.*]] = add i32 [[COMP]], 1
; CHECK-NEXT: [[CROSS:%.*]] = icmp uge i32 [[LAST]], 2
; CHECK-NEXT: br i1 [[CROSS]], label %[[NEXT:.*]], label %[[MERGE:.*]]
; CHECK: [[NEXT]]:
; CHECK: call { i64, i64 } @llvm.dx.resource.load.cbufferrow.2.{{.*}}
; CHECK: [[MERGE]]:
; CHECK-NEXT: [[WINDOW:%.*]] = phi <4 x i64>
; CHECK-COUNT-2: extractelement <4 x i64> [[WINDOW]], i32
; CHECK: ret <2 x i64>
define <2 x i64> @double_width(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 8
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <2 x i64>, ptr addrspace(2) %ptr, align 8
  ret <2 x i64> %value
}

; CHECK-LABEL: define <3 x i16> @half_width(
; CHECK: [[COMP:%.*]] = udiv i32 {{%.*}}, 2
; CHECK: call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}
; CHECK: [[LAST:%.*]] = add i32 [[COMP]], 2
; CHECK-NEXT: [[CROSS:%.*]] = icmp uge i32 [[LAST]], 8
; CHECK-NEXT: br i1 [[CROSS]], label %[[NEXT:.*]], label %[[MERGE:.*]]
; CHECK: [[NEXT]]:
; CHECK: call { i16, i16, i16, i16, i16, i16, i16, i16 } @llvm.dx.resource.load.cbufferrow.8.{{.*}}
; CHECK: [[MERGE]]:
; CHECK-NEXT: [[WINDOW:%.*]] = phi <16 x i16>
; CHECK-COUNT-3: extractelement <16 x i16> [[WINDOW]], i32
; CHECK: ret <3 x i16>
define <3 x i16> @half_width(i1 %cond) {
entry:
  %handle = call target("dx.CBuffer", %CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %CB) %handle, i32 0)
  %offset = select i1 %cond, i32 0, i32 14
  %ptr = getelementptr i8, ptr addrspace(2) %base, i32 %offset
  %value = load <3 x i16>, ptr addrspace(2) %ptr, align 2
  ret <3 x i16> %value
}

; CHECK: [[UNKNOWN]] = !{!"unknown", !"dxil-resource-access"}
!0 = !{!"function_entry_count", i64 10}
