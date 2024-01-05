// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK: declare fastcc void @cconv_fastcc()
// CHECK: declare        void @cconv_ccc()
// CHECK: declare tailcc void @cconv_tailcc()
// CHECK: declare ghccc  void @cconv_ghccc()
llvm.func fastcc @cconv_fastcc()
llvm.func ccc    @cconv_ccc()
llvm.func tailcc @cconv_tailcc()
llvm.func cc_10  @cconv_ghccc()

// CHECK-LABEL: @test_ccs
llvm.func @test_ccs() {
  // CHECK-NEXT: call fastcc void @cconv_fastcc()
  // CHECK-NEXT: call        void @cconv_ccc()
  // CHECK-NEXT: call        void @cconv_ccc()
  // CHECK-NEXT: call tailcc void @cconv_tailcc()
  // CHECK-NEXT: call ghccc  void @cconv_ghccc()
  // CHECK-NEXT: ret void
  llvm.call fastcc @cconv_fastcc() : () -> ()
  llvm.call ccc    @cconv_ccc()    : () -> ()
  llvm.call        @cconv_ccc()    : () -> ()
  llvm.call tailcc @cconv_tailcc() : () -> ()
  llvm.call cc_10  @cconv_ghccc()  : () -> ()
  llvm.return
}

// CHECK-LABEL: @test_ccs_invoke
llvm.func @test_ccs_invoke() attributes { personality = @__gxx_personality_v0 } {
  // CHECK-NEXT: invoke fastcc void @cconv_fastcc()
  // CHECK-NEXT:   to label %[[normal1:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke fastcc @cconv_fastcc() to ^bb1 unwind ^bb6 : () -> ()

^bb1:
  // CHECK: [[normal1]]:
  // CHECK-NEXT: invoke void @cconv_ccc()
  // CHECK-NEXT:   to label %[[normal2:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke ccc @cconv_ccc() to ^bb2 unwind ^bb6 : () -> ()

^bb2:
  // CHECK: [[normal2]]:
  // CHECK-NEXT: invoke void @cconv_ccc()
  // CHECK-NEXT:   to label %[[normal3:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke @cconv_ccc() to ^bb3 unwind ^bb6 : () -> ()

^bb3:
  // CHECK: [[normal3]]:
  // CHECK-NEXT: invoke tailcc void @cconv_tailcc()
  // CHECK-NEXT:   to label %[[normal4:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke tailcc @cconv_tailcc() to ^bb4 unwind ^bb6 : () -> ()

^bb4:
  // CHECK: [[normal4]]:
  // CHECK-NEXT: invoke ghccc void @cconv_ghccc()
  // CHECK-NEXT:   to label %[[normal5:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke cc_10 @cconv_ghccc() to ^bb5 unwind ^bb6 : () -> ()

^bb5:
  // CHECK: [[normal5]]:
  // CHECK-NEXT: ret void
  llvm.return

  // CHECK: [[unwind]]:
  // CHECK-NEXT: landingpad { ptr, i32 }
  // CHECK-NEXT: cleanup
  // CHECK-NEXT: ret void
^bb6:
  %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.return
}
