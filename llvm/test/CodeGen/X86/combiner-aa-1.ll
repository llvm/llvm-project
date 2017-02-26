; RUN: llc < %s --combiner-alias-analysis --combiner-global-alias-analysis
; PR4880

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

%struct.alst_node = type { %struct.node }
%struct.arg_node = type { %struct.node, i8*, %struct.alst_node* }
%struct.arglst_node = type { %struct.alst_node, %struct.arg_node*, %struct.arglst_node* }
%struct.lam_node = type { %struct.alst_node, %struct.arg_node*, %struct.alst_node* }
%struct.node = type { i32 (...)**, %struct.node* }

define i32 @._ZN8lam_node18resolve_name_clashEP8arg_nodeP9alst_node._ZNK8lam_nodeeqERK8exp_node._ZN11arglst_nodeD0Ev(%struct.lam_node* %this.this, %struct.arg_node* %outer_arg, %struct.alst_node* %env.cmp, %struct.arglst_node* %this, i32 %functionID) {
comb_entry:
  %.SV59 = alloca %struct.node*                   ; <%struct.node**> [#uses=1]
  %0 = load i32 (...)**, i32 (...)*** null, align 4            ; <i32 (...)**> [#uses=1]
  %1 = getelementptr inbounds i32 (...)*, i32 (...)** %0, i32 3 ; <i32 (...)**> [#uses=1]
  %2 = load i32 (...)*, i32 (...)** %1, align 4               ; <i32 (...)*> [#uses=1]
  store %struct.node* undef, %struct.node** %.SV59
  %3 = bitcast i32 (...)* %2 to i32 (%struct.node*)* ; <i32 (%struct.node*)*> [#uses=1]
  %4 = tail call i32 %3(%struct.node* undef)      ; <i32> [#uses=0]
  unreachable
}
