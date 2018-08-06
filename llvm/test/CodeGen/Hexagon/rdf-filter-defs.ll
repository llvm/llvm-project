; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s

; Check that this testcase compiles successfully.
; CHECK: dealloc_return

target triple = "hexagon"

%type.0 = type { %type.1, %type.3, i32, i32 }
%type.1 = type { %type.2 }
%type.2 = type { i8 }
%type.3 = type { i8*, [12 x i8] }
%type.4 = type { i8 }

define weak_odr dereferenceable(28) %type.0* @fred(%type.0* %p0, i32 %p1, %type.0* dereferenceable(28) %p2, i32 %p3, i32 %p4) local_unnamed_addr align 2 {
b0:
  %t0 = getelementptr inbounds %type.0, %type.0* %p0, i32 0, i32 2
  %t1 = load i32, i32* %t0, align 4
  %t2 = icmp ult i32 %t1, %p1
  %t3 = getelementptr inbounds %type.0, %type.0* %p2, i32 0, i32 2
  br i1 %t2, label %b2, label %b1

b1:
  %t4 = load i32, i32* %t3, align 4
  %t5 = icmp ult i32 %t4, %p3
  br i1 %t5, label %b2, label %b3

b2:
  %t6 = bitcast %type.0* %p0 to %type.4*
  tail call void @blah(%type.4* %t6)
  %t7 = load i32, i32* %t3, align 4
  %t8 = load i32, i32* %t0, align 4
  br label %b3

b3:
  %t9 = phi i32 [ %t8, %b2 ], [ %t1, %b1 ]
  %t10 = phi i32 [ %t7, %b2 ], [ %t4, %b1 ]
  %t11 = sub i32 %t10, %p3
  %t12 = icmp ult i32 %t11, %p4
  %t13 = select i1 %t12, i32 %t11, i32 %p4
  %t14 = xor i32 %t9, -1
  %t15 = icmp ult i32 %t13, %t14
  br i1 %t15, label %b5, label %b4

b4:
  %t16 = bitcast %type.0* %p0 to %type.4*
  tail call void @danny(%type.4* %t16)
  br label %b5

b5:
  %t17 = icmp eq i32 %t13, 0
  br i1 %t17, label %b33, label %b6

b6:
  %t18 = load i32, i32* %t0, align 4
  %t19 = add i32 %t18, %t13
  %t20 = icmp eq i32 %t19, -1
  br i1 %t20, label %b7, label %b8

b7:
  %t21 = bitcast %type.0* %p0 to %type.4*
  tail call void @danny(%type.4* %t21)
  br label %b8

b8:
  %t22 = getelementptr inbounds %type.0, %type.0* %p0, i32 0, i32 3
  %t23 = load i32, i32* %t22, align 4
  %t24 = icmp ult i32 %t23, %t19
  br i1 %t24, label %b9, label %b10

b9:
  %t25 = load i32, i32* %t0, align 4
  tail call void @sammy(%type.0* nonnull %p0, i32 %t19, i32 %t25)
  %t26 = load i32, i32* %t22, align 4
  br label %b15

b10:
  %t27 = icmp eq i32 %t19, 0
  br i1 %t27, label %b11, label %b15

b11:
  %t28 = icmp ugt i32 %t23, 15
  %t29 = getelementptr inbounds %type.0, %type.0* %p0, i32 0, i32 1
  br i1 %t28, label %b12, label %b13

b12:
  %t30 = getelementptr inbounds %type.3, %type.3* %t29, i32 0, i32 0
  %t31 = load i8*, i8** %t30, align 4
  br label %b14

b13:
  %t32 = bitcast %type.3* %t29 to i8*
  br label %b14

b14:
  %t33 = phi i8* [ %t31, %b12 ], [ %t32, %b13 ]
  store i32 0, i32* %t0, align 4
  br label %b31

b15:
  %t34 = phi i32 [ %t26, %b9 ], [ %t23, %b10 ]
  %t35 = icmp ugt i32 %t34, 15
  %t36 = getelementptr inbounds %type.0, %type.0* %p0, i32 0, i32 1
  br i1 %t35, label %b16, label %b17

b16:
  %t37 = getelementptr inbounds %type.3, %type.3* %t36, i32 0, i32 0
  %t38 = load i8*, i8** %t37, align 4
  br label %b18

b17:
  %t39 = bitcast %type.3* %t36 to i8*
  %t40 = bitcast %type.3* %t36 to i8*
  br label %b18

b18:
  %t41 = phi i8* [ %t38, %b16 ], [ %t39, %b17 ]
  %t42 = phi i8* [ %t38, %b16 ], [ %t40, %b17 ]
  %t43 = getelementptr inbounds i8, i8* %t41, i32 %p1
  %t44 = getelementptr inbounds i8, i8* %t43, i32 %t13
  %t45 = getelementptr inbounds i8, i8* %t42, i32 %p1
  %t46 = load i32, i32* %t0, align 4
  %t47 = sub i32 %t46, %p1
  tail call void @llvm.memmove.p0i8.p0i8.i32(i8* %t44, i8* %t45, i32 %t47, i1 false) #1
  %t48 = icmp eq %type.0* %p0, %p2
  %t49 = load i32, i32* %t22, align 4
  %t50 = icmp ugt i32 %t49, 15
  br i1 %t50, label %b19, label %b20

b19:
  %t51 = getelementptr inbounds %type.3, %type.3* %t36, i32 0, i32 0
  %t52 = load i8*, i8** %t51, align 4
  br label %b21

b20:
  %t53 = bitcast %type.3* %t36 to i8*
  br label %b21

b21:
  %t54 = phi i8* [ %t52, %b19 ], [ %t53, %b20 ]
  %t55 = getelementptr inbounds i8, i8* %t54, i32 %p1
  br i1 %t48, label %b22, label %b26

b22:
  br i1 %t50, label %b23, label %b24

b23:
  %t56 = getelementptr inbounds %type.3, %type.3* %t36, i32 0, i32 0
  %t57 = load i8*, i8** %t56, align 4
  br label %b25

b24:
  %t58 = bitcast %type.3* %t36 to i8*
  br label %b25

b25:
  %t59 = phi i8* [ %t57, %b23 ], [ %t58, %b24 ]
  %t60 = icmp ult i32 %p1, %p3
  %t61 = select i1 %t60, i32 %t13, i32 0
  %t62 = add i32 %t61, %p3
  %t63 = getelementptr inbounds i8, i8* %t59, i32 %t62
  tail call void @llvm.memmove.p0i8.p0i8.i32(i8* %t55, i8* %t63, i32 %t13, i1 false) #1
  br label %b27

b26:
  %t64 = getelementptr inbounds %type.0, %type.0* %p2, i32 0, i32 3
  %t65 = load i32, i32* %t64, align 4
  %t66 = icmp ugt i32 %t65, 15
  %t67 = getelementptr inbounds %type.0, %type.0* %p2, i32 0, i32 1
  %t68 = getelementptr inbounds %type.3, %type.3* %t67, i32 0, i32 0
  %t69 = load i8*, i8** %t68, align 4
  %t70 = bitcast %type.3* %t67 to i8*
  %t71 = select i1 %t66, i8* %t69, i8* %t70
  %t72 = getelementptr inbounds i8, i8* %t71, i32 %p3
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %t55, i8* %t72, i32 %t13, i1 false) #1
  br label %b27

b27:
  %t73 = load i32, i32* %t22, align 4
  %t74 = icmp ugt i32 %t73, 15
  br i1 %t74, label %b28, label %b29

b28:
  %t75 = getelementptr inbounds %type.3, %type.3* %t36, i32 0, i32 0
  %t76 = load i8*, i8** %t75, align 4
  br label %b30

b29:
  %t77 = bitcast %type.3* %t36 to i8*
  br label %b30

b30:
  %t78 = phi i8* [ %t76, %b28 ], [ %t77, %b29 ]
  store i32 %t19, i32* %t0, align 4
  %t79 = getelementptr inbounds i8, i8* %t78, i32 %t19
  br label %b31

b31:
  %t80 = phi i8* [ %t33, %b14 ], [ %t79, %b30 ]
  store i8 0, i8* %t80, align 1
  br label %b33

b33:
  ret %type.0* %p0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #0
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) #0

declare void @blah(%type.4*) local_unnamed_addr
declare void @danny(%type.4*) local_unnamed_addr
declare void @sammy(%type.0*, i32, i32) local_unnamed_addr align 2

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind }
