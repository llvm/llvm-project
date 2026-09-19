; RUN: opt -S -dxil-resource-type -dxil-resource-access -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

%struct.S = type { i32, i32 }
%__cblayout_CB = type <{ <3 x i32>, target("dx.Padding", 4) }>
%__cblayout_CB2 = type <{ <{ [9 x <{ <3 x i32>, target("dx.Padding", 4) }>], <3 x i32> }> }>


@OutArr.str = internal unnamed_addr constant [7 x i8] c"OutArr\00"
@CBArr.str = internal unnamed_addr constant [6 x i8] c"CBArr\00"
@CB.str = private unnamed_addr constant [3 x i8] c"CB\00"

; CHECK-LABEL: define i32 @gep_phi(
; CHECK-SAME: i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]], i32 %[[X:.*]], i32 %[[Y:.*]])
define i32 @gep_phi(i1 %cond, i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
; CHECK: entry:
; CHECK-NEXT: %[[X_OFFSET:.*]] = mul i32 %[[X]], 8
; CHECK-NEXT: %[[X_FIELD:.*]] = add i32 4, %[[X_OFFSET]]
; CHECK-NEXT: br i1 %[[COND]], label %then, label %merge
  %handle0 = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 -1, i32 0, ptr @OutArr.str)
  %ptr0 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", %struct.S, 1, 0) %handle0, i32 %a)
  %field0 = getelementptr %struct.S, ptr %ptr0, i32 %x, i32 1
  br i1 %cond, label %then, label %merge

then:
; CHECK: then:
; CHECK-NEXT: %[[Y_OFFSET:.*]] = mul i32 %[[Y]], 8
; CHECK-NEXT: %[[Y_FIELD:.*]] = add i32 4, %[[Y_OFFSET]]
; CHECK-NEXT: br label %merge
  %handle1 = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 -1, i32 1, ptr @OutArr.str)
  %ptr1 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", %struct.S, 1, 0) %handle1, i32 %b)
  %field1 = getelementptr %struct.S, ptr %ptr1, i32 %y, i32 1
  br label %merge

merge:
; CHECK: merge:
; CHECK-NEXT: %[[ELEMENT:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %then ]
; CHECK-NEXT: %[[HANDLE_IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %then ]
; CHECK-NEXT: %[[OFFSET:.*]] = phi i32 [ %[[X_FIELD]], %entry ], [ %[[Y_FIELD]], %then ]
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_s_struct.Ss_1_0t(i32 2, i32 0, i32 -1, i32 %[[HANDLE_IDX]], ptr @OutArr.str)
; CHECK-NEXT: %[[LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_s_struct.Ss_1_0t(target("dx.RawBuffer", %struct.S, 1, 0) %[[HANDLE]], i32 %[[ELEMENT]], i32 %[[OFFSET]])
; CHECK-NEXT: %[[VALUE:.*]] = extractvalue { i32, i1 } %[[LOAD]], 0
; CHECK-NEXT: ret i32 %[[VALUE]]
  %ptr = phi ptr [ %field0, %entry ], [ %field1, %then ]
  %value = load i32, ptr %ptr
  ret i32 %value
}

; CHECK-LABEL: define i32 @gep_phi_one_branch(
; CHECK-SAME: i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define i32 @gep_phi_one_branch(i1 %cond, i32 %a, i32 %b) {
entry:
  %handle0 = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 -1, i32 0, ptr @OutArr.str)
  %ptr0 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", %struct.S, 1, 0) %handle0, i32 %a)
; CHECK: br i1 %[[COND]], label %then, label %merge
  br i1 %cond, label %then, label %merge

then:
  %handle1 = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 -1, i32 1, ptr @OutArr.str)
  %ptr1 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", %struct.S, 1, 0) %handle1, i32 %b)
  %field1 = getelementptr %struct.S, ptr %ptr1, i32 0, i32 1
  ; CHECK: br label %merge
  br label %merge

merge:
; CHECK: merge:
; CHECK-NEXT: %[[ELEMENT:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %then ]
; CHECK-NEXT: %[[HANDLE_IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %then ]
; CHECK-NEXT: %[[OFFSET:.*]] = phi i32 [ 0, %entry ], [ 4, %then ]
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_s_struct.Ss_1_0t(i32 2, i32 0, i32 -1, i32 %[[HANDLE_IDX]], ptr @OutArr.str)
; CHECK-NEXT: %[[LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_s_struct.Ss_1_0t(target("dx.RawBuffer", %struct.S, 1, 0) %[[HANDLE]], i32 %[[ELEMENT]], i32 %[[OFFSET]])
; CHECK-NEXT: %[[VALUE:.*]] = extractvalue { i32, i1 } %[[LOAD]], 0
; CHECK-NEXT: ret i32 %[[VALUE]]
  %ptr = phi ptr [ %ptr0, %entry ], [ %field1, %then ]
  %value = load i32, ptr %ptr
  ret i32 %value
}

; CHECK-LABEL: define i32 @gep_select_one_branch(
; CHECK-SAME: i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define i32 @gep_select_one_branch(i1 %cond, i32 %a, i32 %b) {
entry:
; CHECK: entry:
; CHECK-NEXT: %[[ELEMENT:.*]] = select i1 %[[COND]], i32 %[[A]], i32 %[[B]]
; CHECK-NEXT: %[[HANDLE_IDX:.*]] = select i1 %[[COND]], i32 0, i32 1
; CHECK-NEXT: %[[OFFSET:.*]] = select i1 %[[COND]], i32 0, i32 4
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_s_struct.Ss_1_0t(i32 2, i32 0, i32 -1, i32 %[[HANDLE_IDX]], ptr @OutArr.str)
; CHECK-NEXT: %[[LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_s_struct.Ss_1_0t(target("dx.RawBuffer", %struct.S, 1, 0) %[[HANDLE]], i32 %[[ELEMENT]], i32 %[[OFFSET]])
; CHECK-NEXT: %[[VALUE:.*]] = extractvalue { i32, i1 } %[[LOAD]], 0
; CHECK-NEXT: ret i32 %[[VALUE]]
  %handle0 = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 -1, i32 0, ptr @OutArr.str)
  %ptr0 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", %struct.S, 1, 0) %handle0, i32 %a)
  %handle1 = call target("dx.RawBuffer", %struct.S, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 -1, i32 1, ptr @OutArr.str)
  %ptr1 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", %struct.S, 1, 0) %handle1, i32 %b)
  %field1 = getelementptr %struct.S, ptr %ptr1, i32 0, i32 1
  %ptr = select i1 %cond, ptr %ptr0, ptr %field1
  %value = load i32, ptr %ptr
  ret i32 %value
}

; CHECK-LABEL: define <3 x i32> @basepointer_phi(
; CHECK-SAME: i1 %[[COND:.*]])
define <3 x i32> @basepointer_phi(i1 %cond) {
entry:
  %handle0 = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 0, ptr @CBArr.str)
  %ptr0 = call ptr addrspace(2) @llvm.dx.resource.getbasepointer(target("dx.CBuffer", %__cblayout_CB) %handle0)
; CHECK: br i1 %[[COND]], label %then, label %merge
  br i1 %cond, label %then, label %merge

then:
  %handle1 = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 1, ptr @CBArr.str)
  %ptr1 = call ptr addrspace(2) @llvm.dx.resource.getbasepointer(target("dx.CBuffer", %__cblayout_CB) %handle1)
; CHECK: br label %merge
  br label %merge

merge:
; CHECK: merge:
; CHECK-NEXT: %[[HANDLE_IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %then ]
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst(i32 1, i32 0, i32 2, i32 %[[HANDLE_IDX]], ptr @CBArr.str)
; CHECK-NEXT: %[[LOAD:.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s___cblayout_CBst(target("dx.CBuffer", %__cblayout_CB) %[[HANDLE]], i32 1)
; CHECK-NEXT: %[[X:.*]] = extractvalue { i32, i32, i32, i32 } %[[LOAD]], 0
; CHECK-NEXT: %[[Y:.*]] = extractvalue { i32, i32, i32, i32 } %[[LOAD]], 1
; CHECK-NEXT: %[[Z:.*]] = extractvalue { i32, i32, i32, i32 } %[[LOAD]], 2
; CHECK-NEXT: %[[V0:.*]] = insertelement <3 x i32> poison, i32 %[[X]], i32 0
; CHECK-NEXT: %[[V1:.*]] = insertelement <3 x i32> %[[V0]], i32 %[[Y]], i32 1
; CHECK-NEXT: %[[V2:.*]] = insertelement <3 x i32> %[[V1]], i32 %[[Z]], i32 2
; CHECK-NEXT: ret <3 x i32> %[[V2]]
  %ptr = phi ptr addrspace(2) [ %ptr0, %entry ], [ %ptr1, %then ]
  %row1 = getelementptr i8, ptr addrspace(2) %ptr, i32 16
  %value = load <3 x i32>, ptr addrspace(2) %row1, align 16
  ret <3 x i32> %value
}

; CHECK-LABEL: define <3 x i32> @basepointer_phi_variable_row(
; CHECK-SAME: i1 %[[COND:.*]], i32 %[[ROW:.*]])
define <3 x i32> @basepointer_phi_variable_row(i1 %cond, i32 %row) {
entry:
  %handle0 = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 0, ptr @CBArr.str)
  %ptr0 = call ptr addrspace(2) @llvm.dx.resource.getbasepointer(target("dx.CBuffer", %__cblayout_CB) %handle0)
; CHECK: br i1 %[[COND]], label %then, label %merge
  br i1 %cond, label %then, label %merge

then:
  %handle1 = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 1, ptr @CBArr.str)
  %ptr1 = call ptr addrspace(2) @llvm.dx.resource.getbasepointer(target("dx.CBuffer", %__cblayout_CB) %handle1)
; CHECK: br label %merge
  br label %merge

merge:
; CHECK: merge:
; CHECK-NEXT: %[[HANDLE_IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %then ]
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst(i32 1, i32 0, i32 2, i32 %[[HANDLE_IDX]], ptr @CBArr.str)
; CHECK-NEXT: %[[LOAD:.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s___cblayout_CBst(target("dx.CBuffer", %__cblayout_CB) %[[HANDLE]], i32 %[[ROW]])
  %ptr = phi ptr addrspace(2) [ %ptr0, %entry ], [ %ptr1, %then ]
  %rowptr = getelementptr %__cblayout_CB, ptr addrspace(2) %ptr, i32 %row
  %value = load <3 x i32>, ptr addrspace(2) %rowptr, align 16
  ret <3 x i32> %value
}

; CHECK-LABEL: define void @cb_phi_handle(
; CHECK-SAME: ptr %[[DST:.*]], i1 %[[COND:.*]], i32 %[[IDX:.*]])
define void @cb_phi_handle(ptr %dst, i1 %cond, i32 %idx) {
entry:
  %h0 = call target("dx.CBuffer", %__cblayout_CB2) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 0, ptr @CB.str)
  %p0 = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB2) %h0, i32 0)
  br i1 %cond, label %if.then, label %main

if.then:
  %h1 = call target("dx.CBuffer", %__cblayout_CB2) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 1, ptr @CB.str)
  %p1 = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB2) %h1, i32 0)
  br label %main

main:
; CHECK: main:
; CHECK-NEXT: %[[HANDLE_IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %if.then ]
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.CBuffer", %__cblayout_CB2) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CB2st(i32 1, i32 0, i32 2, i32 %[[HANDLE_IDX]], ptr @CB.str)
; CHECK-NEXT: %[[LOAD:.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.i32.i32.i32.i32.tdx.CBuffer_s___cblayout_CB2st(target("dx.CBuffer", %__cblayout_CB2) %[[HANDLE]], i32 %[[IDX]])
  %pp = phi ptr addrspace(2) [ %p0, %entry ], [ %p1, %if.then ]
  %gep = getelementptr <{ <3 x i32>, target("dx.Padding", 4) }>, ptr addrspace(2) %pp, i32 %idx
  %ld = load <3 x i32>, ptr addrspace(2) %gep, align 16
  %e = extractelement <3 x i32> %ld, i32 0
  store i32 %e, ptr %dst, align 4
  ret void
}
