// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @clusterlaunchcontrol_try_cancel(%addr: !llvm.ptr<3>, %mbar: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @clusterlaunchcontrol_try_cancel(ptr addrspace(3) %0, ptr addrspace(3) %1) {
  // CHECK-NEXT: call void @llvm.nvvm.clusterlaunchcontrol.try_cancel.async.shared(ptr addrspace(3) %0, ptr addrspace(3) %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.clusterlaunchcontrol.try.cancel %addr, %mbar
  llvm.return
}

llvm.func @clusterlaunchcontrol_try_cancel_multicast(%addr: !llvm.ptr<3>, %mbar: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @clusterlaunchcontrol_try_cancel_multicast(ptr addrspace(3) %0, ptr addrspace(3) %1) {
  // CHECK-NEXT: call void @llvm.nvvm.clusterlaunchcontrol.try_cancel.async.multicast.shared(ptr addrspace(3) %0, ptr addrspace(3) %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.clusterlaunchcontrol.try.cancel multicast, %addr, %mbar
  llvm.return
}

llvm.func @clusterlaunchcontrol_query_cancel_is_canceled(%try_cancel_response: i128) {
  // CHECK-LABEL: define void @clusterlaunchcontrol_query_cancel_is_canceled(i128 %0) {
  // CHECK-NEXT: %2 = call i1 @llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled(i128 %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %res = nvvm.clusterlaunchcontrol.query.cancel query = is_canceled, %try_cancel_response : i1
  llvm.return
}

llvm.func @clusterlaunchcontrol_query_cancel_get_first_cta_id_x(%try_cancel_response: i128) {
  // CHECK-LABEL: define void @clusterlaunchcontrol_query_cancel_get_first_cta_id_x(i128 %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x(i128 %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %res = nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_x, %try_cancel_response : i32
  llvm.return
}

llvm.func @clusterlaunchcontrol_query_cancel_get_first_cta_id_y(%try_cancel_response: i128) {
  // CHECK-LABEL: define void @clusterlaunchcontrol_query_cancel_get_first_cta_id_y(i128 %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y(i128 %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %res = nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_y, %try_cancel_response : i32
  llvm.return
}

llvm.func @clusterlaunchcontrol_query_cancel_get_first_cta_id_z(%try_cancel_response: i128) {
  // CHECK-LABEL: define void @clusterlaunchcontrol_query_cancel_get_first_cta_id_z(i128 %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z(i128 %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %res = nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_z, %try_cancel_response : i32
  llvm.return
}
