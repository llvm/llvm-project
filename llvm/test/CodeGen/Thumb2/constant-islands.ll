; RUN: llc < %s -mtriple=arm-apple-ios   -mcpu=cortex-a8 -O0 -filetype=obj -verify-machine-dom-info -o %t.o
; RUN: llc < %s -mtriple=thumb-apple-ios -mcpu=cortex-a8 -O0 -filetype=obj -o %t.o
; RUN: llc < %s -mtriple=arm-apple-ios   -mcpu=cortex-a8 -O2 -filetype=obj -verify-machineinstrs -o %t.o
; RUN: llc < %s -mtriple=thumb-apple-ios -mcpu=cortex-a8 -O2 -filetype=obj -verify-machineinstrs -o %t.o
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

; This function comes from the Bullet test.  It is quite big, and exercises the
; constant island pass a bit.  It has caused failures, including
; <rdar://problem/10670199>
;
; It is unlikely that this code will continue to create the exact conditions
; that broke the arm constant island pass in the past, but it is still useful to
; force the pass to split basic blocks etc.
;
; The run lines above force the integrated assembler to be enabled so it can
; catch any illegal displacements.  Other than that, we depend on the constant
; island pass assertions.

%class.btVector3 = type { [4 x float] }
%class.btTransform = type { %class.btMatrix3x3, %class.btVector3 }
%class.btMatrix3x3 = type { [3 x %class.btVector3] }
%class.btCapsuleShape = type { %class.btConvexInternalShape, i32 }
%class.btConvexInternalShape = type { %class.btConvexShape, %class.btVector3, %class.btVector3, float, float }
%class.btConvexShape = type { %class.btCollisionShape }
%class.btCollisionShape = type { ptr, i32, ptr }
%class.RagDoll = type { ptr, ptr, [11 x ptr], [11 x ptr], [10 x ptr] }
%class.btDynamicsWorld = type { %class.btCollisionWorld, ptr, ptr, ptr, %struct.btContactSolverInfo }
%class.btCollisionWorld = type { ptr, %class.btAlignedObjectArray, ptr, %struct.btDispatcherInfo, ptr, ptr, ptr, i8 }
%class.btAlignedObjectArray = type { %class.btAlignedAllocator, i32, i32, ptr, i8 }
%class.btAlignedAllocator = type { i8 }
%class.btCollisionObject = type { ptr, %class.btTransform, %class.btTransform, %class.btVector3, %class.btVector3, %class.btVector3, i8, float, ptr, ptr, ptr, i32, i32, i32, i32, float, float, float, ptr, i32, float, float, float, i8, [7 x i8] }
%struct.btBroadphaseProxy = type { ptr, i16, i16, ptr, i32, %class.btVector3, %class.btVector3 }
%class.btDispatcher = type { ptr }
%struct.btDispatcherInfo = type { float, i32, i32, float, i8, ptr, i8, i8, i8, float, i8, float, ptr }
%class.btIDebugDraw = type { ptr }
%class.btStackAlloc = type opaque
%class.btBroadphaseInterface = type { ptr }
%struct.btContactSolverInfo = type { %struct.btContactSolverInfoData }
%struct.btContactSolverInfoData = type { float, float, float, float, float, i32, float, float, float, float, float, i32, float, float, float, i32, i32 }
%class.btRigidBody = type { %class.btCollisionObject, %class.btMatrix3x3, %class.btVector3, %class.btVector3, float, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, float, float, i8, float, float, float, float, float, float, ptr, %class.btAlignedObjectArray.22, i32, i32, i32 }
%class.btMotionState = type { ptr }
%class.btAlignedObjectArray.22 = type { %class.btAlignedAllocator.23, i32, i32, ptr, i8 }
%class.btAlignedAllocator.23 = type { i8 }
%class.btTypedConstraint = type { ptr, %struct.btTypedObject, i32, i32, i8, ptr, ptr, float, float, %class.btVector3, %class.btVector3, %class.btVector3 }
%struct.btTypedObject = type { i32 }
%class.btHingeConstraint = type { %class.btTypedConstraint, [3 x %class.btJacobianEntry], [3 x %class.btJacobianEntry], %class.btTransform, %class.btTransform, float, float, float, float, float, float, float, float, float, float, float, float, float, i8, i8, i8, i8, i8, float }
%class.btJacobianEntry = type { %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, float }
%class.btConeTwistConstraint = type { %class.btTypedConstraint, [3 x %class.btJacobianEntry], %class.btTransform, %class.btTransform, float, float, float, float, float, float, float, float, %class.btVector3, %class.btVector3, float, float, float, float, float, float, float, float, i8, i8, i8, i8, float, float, %class.btVector3, i8, i8, %class.btQuaternion, float, %class.btVector3 }
%class.btQuaternion = type { %class.btQuadWord }
%class.btQuadWord = type { [4 x float] }

@_ZTV7RagDoll = external unnamed_addr constant [4 x ptr]

declare noalias ptr @_Znwm(i32)

declare i32 @__gxx_personality_sj0(...)

declare void @_ZdlPv(ptr) nounwind

declare ptr @_ZN9btVector3C1ERKfS1_S1_(ptr, ptr, ptr, ptr) unnamed_addr inlinehint ssp align 2

declare void @_ZSt9terminatev()

declare ptr @_ZN11btTransformC1Ev(ptr) unnamed_addr ssp align 2

declare void @_ZN11btTransform11setIdentityEv(ptr) ssp align 2

declare void @_ZN11btTransform9setOriginERK9btVector3(ptr, ptr) nounwind inlinehint ssp align 2

declare ptr @_ZN13btConvexShapenwEm(i32) inlinehint ssp align 2

declare void @_ZN13btConvexShapedlEPv(ptr) inlinehint ssp align 2

declare ptr @_ZN14btCapsuleShapeC1Eff(ptr, float, float)

declare ptr @_ZN11btTransform8getBasisEv(ptr) nounwind inlinehint ssp align 2

define ptr @_ZN7RagDollC2EP15btDynamicsWorldRK9btVector3f(ptr %this, ptr %ownerWorld, ptr %positionOffset, float %scale) unnamed_addr ssp align 2 personality ptr @__gxx_personality_sj0 {
entry:
  %retval = alloca ptr, align 4
  %this.addr = alloca ptr, align 4
  %ownerWorld.addr = alloca ptr, align 4
  %positionOffset.addr = alloca ptr, align 4
  %scale.addr = alloca float, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  %offset = alloca %class.btTransform, align 4
  %transform = alloca %class.btTransform, align 4
  %ref.tmp = alloca %class.btVector3, align 4
  %ref.tmp97 = alloca %class.btVector3, align 4
  %ref.tmp98 = alloca float, align 4
  %ref.tmp99 = alloca float, align 4
  %ref.tmp100 = alloca float, align 4
  %ref.tmp102 = alloca %class.btTransform, align 4
  %ref.tmp107 = alloca %class.btVector3, align 4
  %ref.tmp108 = alloca %class.btVector3, align 4
  %ref.tmp109 = alloca float, align 4
  %ref.tmp110 = alloca float, align 4
  %ref.tmp111 = alloca float, align 4
  %ref.tmp113 = alloca %class.btTransform, align 4
  %ref.tmp119 = alloca %class.btVector3, align 4
  %ref.tmp120 = alloca %class.btVector3, align 4
  %ref.tmp121 = alloca float, align 4
  %ref.tmp122 = alloca float, align 4
  %ref.tmp123 = alloca float, align 4
  %ref.tmp125 = alloca %class.btTransform, align 4
  %ref.tmp131 = alloca %class.btVector3, align 4
  %ref.tmp132 = alloca %class.btVector3, align 4
  %ref.tmp133 = alloca float, align 4
  %ref.tmp134 = alloca float, align 4
  %ref.tmp135 = alloca float, align 4
  %ref.tmp137 = alloca %class.btTransform, align 4
  %ref.tmp143 = alloca %class.btVector3, align 4
  %ref.tmp144 = alloca %class.btVector3, align 4
  %ref.tmp145 = alloca float, align 4
  %ref.tmp146 = alloca float, align 4
  %ref.tmp147 = alloca float, align 4
  %ref.tmp149 = alloca %class.btTransform, align 4
  %ref.tmp155 = alloca %class.btVector3, align 4
  %ref.tmp156 = alloca %class.btVector3, align 4
  %ref.tmp157 = alloca float, align 4
  %ref.tmp158 = alloca float, align 4
  %ref.tmp159 = alloca float, align 4
  %ref.tmp161 = alloca %class.btTransform, align 4
  %ref.tmp167 = alloca %class.btVector3, align 4
  %ref.tmp168 = alloca %class.btVector3, align 4
  %ref.tmp169 = alloca float, align 4
  %ref.tmp170 = alloca float, align 4
  %ref.tmp171 = alloca float, align 4
  %ref.tmp173 = alloca %class.btTransform, align 4
  %ref.tmp179 = alloca %class.btVector3, align 4
  %ref.tmp180 = alloca %class.btVector3, align 4
  %ref.tmp181 = alloca float, align 4
  %ref.tmp182 = alloca float, align 4
  %ref.tmp183 = alloca float, align 4
  %ref.tmp186 = alloca %class.btTransform, align 4
  %ref.tmp192 = alloca %class.btVector3, align 4
  %ref.tmp193 = alloca %class.btVector3, align 4
  %ref.tmp194 = alloca float, align 4
  %ref.tmp195 = alloca float, align 4
  %ref.tmp196 = alloca float, align 4
  %ref.tmp199 = alloca %class.btTransform, align 4
  %ref.tmp205 = alloca %class.btVector3, align 4
  %ref.tmp206 = alloca %class.btVector3, align 4
  %ref.tmp207 = alloca float, align 4
  %ref.tmp208 = alloca float, align 4
  %ref.tmp209 = alloca float, align 4
  %ref.tmp212 = alloca %class.btTransform, align 4
  %ref.tmp218 = alloca %class.btVector3, align 4
  %ref.tmp219 = alloca %class.btVector3, align 4
  %ref.tmp220 = alloca float, align 4
  %ref.tmp221 = alloca float, align 4
  %ref.tmp222 = alloca float, align 4
  %ref.tmp225 = alloca %class.btTransform, align 4
  %i = alloca i32, align 4
  %hingeC = alloca ptr, align 4
  %coneC = alloca ptr, align 4
  %localA = alloca %class.btTransform, align 4
  %localB = alloca %class.btTransform, align 4
  %ref.tmp240 = alloca %class.btVector3, align 4
  %ref.tmp241 = alloca %class.btVector3, align 4
  %ref.tmp242 = alloca float, align 4
  %ref.tmp243 = alloca float, align 4
  %ref.tmp244 = alloca float, align 4
  %ref.tmp247 = alloca %class.btVector3, align 4
  %ref.tmp248 = alloca %class.btVector3, align 4
  %ref.tmp249 = alloca float, align 4
  %ref.tmp250 = alloca float, align 4
  %ref.tmp251 = alloca float, align 4
  %ref.tmp266 = alloca %class.btVector3, align 4
  %ref.tmp267 = alloca %class.btVector3, align 4
  %ref.tmp268 = alloca float, align 4
  %ref.tmp269 = alloca float, align 4
  %ref.tmp270 = alloca float, align 4
  %ref.tmp273 = alloca %class.btVector3, align 4
  %ref.tmp274 = alloca %class.btVector3, align 4
  %ref.tmp275 = alloca float, align 4
  %ref.tmp276 = alloca float, align 4
  %ref.tmp277 = alloca float, align 4
  %ref.tmp295 = alloca %class.btVector3, align 4
  %ref.tmp296 = alloca %class.btVector3, align 4
  %ref.tmp297 = alloca float, align 4
  %ref.tmp298 = alloca float, align 4
  %ref.tmp299 = alloca float, align 4
  %ref.tmp302 = alloca %class.btVector3, align 4
  %ref.tmp303 = alloca %class.btVector3, align 4
  %ref.tmp304 = alloca float, align 4
  %ref.tmp305 = alloca float, align 4
  %ref.tmp306 = alloca float, align 4
  %ref.tmp324 = alloca %class.btVector3, align 4
  %ref.tmp325 = alloca %class.btVector3, align 4
  %ref.tmp326 = alloca float, align 4
  %ref.tmp327 = alloca float, align 4
  %ref.tmp328 = alloca float, align 4
  %ref.tmp331 = alloca %class.btVector3, align 4
  %ref.tmp332 = alloca %class.btVector3, align 4
  %ref.tmp333 = alloca float, align 4
  %ref.tmp334 = alloca float, align 4
  %ref.tmp335 = alloca float, align 4
  %ref.tmp353 = alloca %class.btVector3, align 4
  %ref.tmp354 = alloca %class.btVector3, align 4
  %ref.tmp355 = alloca float, align 4
  %ref.tmp356 = alloca float, align 4
  %ref.tmp357 = alloca float, align 4
  %ref.tmp360 = alloca %class.btVector3, align 4
  %ref.tmp361 = alloca %class.btVector3, align 4
  %ref.tmp362 = alloca float, align 4
  %ref.tmp363 = alloca float, align 4
  %ref.tmp364 = alloca float, align 4
  %ref.tmp382 = alloca %class.btVector3, align 4
  %ref.tmp383 = alloca %class.btVector3, align 4
  %ref.tmp384 = alloca float, align 4
  %ref.tmp385 = alloca float, align 4
  %ref.tmp386 = alloca float, align 4
  %ref.tmp389 = alloca %class.btVector3, align 4
  %ref.tmp390 = alloca %class.btVector3, align 4
  %ref.tmp391 = alloca float, align 4
  %ref.tmp392 = alloca float, align 4
  %ref.tmp393 = alloca float, align 4
  %ref.tmp411 = alloca %class.btVector3, align 4
  %ref.tmp412 = alloca %class.btVector3, align 4
  %ref.tmp413 = alloca float, align 4
  %ref.tmp414 = alloca float, align 4
  %ref.tmp415 = alloca float, align 4
  %ref.tmp418 = alloca %class.btVector3, align 4
  %ref.tmp419 = alloca %class.btVector3, align 4
  %ref.tmp420 = alloca float, align 4
  %ref.tmp421 = alloca float, align 4
  %ref.tmp422 = alloca float, align 4
  %ref.tmp440 = alloca %class.btVector3, align 4
  %ref.tmp441 = alloca %class.btVector3, align 4
  %ref.tmp442 = alloca float, align 4
  %ref.tmp443 = alloca float, align 4
  %ref.tmp444 = alloca float, align 4
  %ref.tmp447 = alloca %class.btVector3, align 4
  %ref.tmp448 = alloca %class.btVector3, align 4
  %ref.tmp449 = alloca float, align 4
  %ref.tmp450 = alloca float, align 4
  %ref.tmp451 = alloca float, align 4
  %ref.tmp469 = alloca %class.btVector3, align 4
  %ref.tmp470 = alloca %class.btVector3, align 4
  %ref.tmp471 = alloca float, align 4
  %ref.tmp472 = alloca float, align 4
  %ref.tmp473 = alloca float, align 4
  %ref.tmp476 = alloca %class.btVector3, align 4
  %ref.tmp477 = alloca %class.btVector3, align 4
  %ref.tmp478 = alloca float, align 4
  %ref.tmp479 = alloca float, align 4
  %ref.tmp480 = alloca float, align 4
  %ref.tmp498 = alloca %class.btVector3, align 4
  %ref.tmp499 = alloca %class.btVector3, align 4
  %ref.tmp500 = alloca float, align 4
  %ref.tmp501 = alloca float, align 4
  %ref.tmp502 = alloca float, align 4
  %ref.tmp505 = alloca %class.btVector3, align 4
  %ref.tmp506 = alloca %class.btVector3, align 4
  %ref.tmp507 = alloca float, align 4
  %ref.tmp508 = alloca float, align 4
  %ref.tmp509 = alloca float, align 4
  store ptr %this, ptr %this.addr, align 4
  store ptr %ownerWorld, ptr %ownerWorld.addr, align 4
  store ptr %positionOffset, ptr %positionOffset.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  %this1 = load ptr, ptr %this.addr
  store ptr %this1, ptr %retval
  store ptr getelementptr inbounds ([4 x ptr], ptr @_ZTV7RagDoll, i64 0, i64 2), ptr %this1
  %m_ownerWorld = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %0 = load ptr, ptr %ownerWorld.addr, align 4
  store ptr %0, ptr %m_ownerWorld, align 4
  %call = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %1 = load float, ptr %scale.addr, align 4
  %mul = fmul float 0x3FC3333340000000, %1
  %2 = load float, ptr %scale.addr, align 4
  %mul2 = fmul float 0x3FC99999A0000000, %2
  %call3 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call, float %mul, float %mul2)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %m_shapes = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  store ptr %call, ptr %m_shapes, align 4
  %call5 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %3 = load float, ptr %scale.addr, align 4
  %mul6 = fmul float 0x3FC3333340000000, %3
  %4 = load float, ptr %scale.addr, align 4
  %mul7 = fmul float 0x3FD1EB8520000000, %4
  %call10 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call5, float %mul6, float %mul7)
          to label %invoke.cont9 unwind label %lpad8

invoke.cont9:                                     ; preds = %invoke.cont
  %m_shapes12 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx13 = getelementptr inbounds [11 x ptr], ptr %m_shapes12, i32 0, i32 1
  store ptr %call5, ptr %arrayidx13, align 4
  %call14 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %5 = load float, ptr %scale.addr, align 4
  %mul15 = fmul float 0x3FB99999A0000000, %5
  %6 = load float, ptr %scale.addr, align 4
  %mul16 = fmul float 0x3FA99999A0000000, %6
  %call19 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call14, float %mul15, float %mul16)
          to label %invoke.cont18 unwind label %lpad17

invoke.cont18:                                    ; preds = %invoke.cont9
  %m_shapes21 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx22 = getelementptr inbounds [11 x ptr], ptr %m_shapes21, i32 0, i32 2
  store ptr %call14, ptr %arrayidx22, align 4
  %call23 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %7 = load float, ptr %scale.addr, align 4
  %mul24 = fmul float 0x3FB1EB8520000000, %7
  %8 = load float, ptr %scale.addr, align 4
  %mul25 = fmul float 0x3FDCCCCCC0000000, %8
  %call28 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call23, float %mul24, float %mul25)
          to label %invoke.cont27 unwind label %lpad26

invoke.cont27:                                    ; preds = %invoke.cont18
  %m_shapes30 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx31 = getelementptr inbounds [11 x ptr], ptr %m_shapes30, i32 0, i32 3
  store ptr %call23, ptr %arrayidx31, align 4
  %call32 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %9 = load float, ptr %scale.addr, align 4
  %mul33 = fmul float 0x3FA99999A0000000, %9
  %10 = load float, ptr %scale.addr, align 4
  %mul34 = fmul float 0x3FD7AE1480000000, %10
  %call37 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call32, float %mul33, float %mul34)
          to label %invoke.cont36 unwind label %lpad35

invoke.cont36:                                    ; preds = %invoke.cont27
  %m_shapes39 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx40 = getelementptr inbounds [11 x ptr], ptr %m_shapes39, i32 0, i32 4
  store ptr %call32, ptr %arrayidx40, align 4
  %call41 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %11 = load float, ptr %scale.addr, align 4
  %mul42 = fmul float 0x3FB1EB8520000000, %11
  %12 = load float, ptr %scale.addr, align 4
  %mul43 = fmul float 0x3FDCCCCCC0000000, %12
  %call46 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call41, float %mul42, float %mul43)
          to label %invoke.cont45 unwind label %lpad44

invoke.cont45:                                    ; preds = %invoke.cont36
  %m_shapes48 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx49 = getelementptr inbounds [11 x ptr], ptr %m_shapes48, i32 0, i32 5
  store ptr %call41, ptr %arrayidx49, align 4
  %call50 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %13 = load float, ptr %scale.addr, align 4
  %mul51 = fmul float 0x3FA99999A0000000, %13
  %14 = load float, ptr %scale.addr, align 4
  %mul52 = fmul float 0x3FD7AE1480000000, %14
  %call55 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call50, float %mul51, float %mul52)
          to label %invoke.cont54 unwind label %lpad53

invoke.cont54:                                    ; preds = %invoke.cont45
  %m_shapes57 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx58 = getelementptr inbounds [11 x ptr], ptr %m_shapes57, i32 0, i32 6
  store ptr %call50, ptr %arrayidx58, align 4
  %call59 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %15 = load float, ptr %scale.addr, align 4
  %mul60 = fmul float 0x3FA99999A0000000, %15
  %16 = load float, ptr %scale.addr, align 4
  %mul61 = fmul float 0x3FD51EB860000000, %16
  %call64 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call59, float %mul60, float %mul61)
          to label %invoke.cont63 unwind label %lpad62

invoke.cont63:                                    ; preds = %invoke.cont54
  %m_shapes66 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx67 = getelementptr inbounds [11 x ptr], ptr %m_shapes66, i32 0, i32 7
  store ptr %call59, ptr %arrayidx67, align 4
  %call68 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %17 = load float, ptr %scale.addr, align 4
  %mul69 = fmul float 0x3FA47AE140000000, %17
  %18 = load float, ptr %scale.addr, align 4
  %mul70 = fmul float 2.500000e-01, %18
  %call73 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call68, float %mul69, float %mul70)
          to label %invoke.cont72 unwind label %lpad71

invoke.cont72:                                    ; preds = %invoke.cont63
  %m_shapes75 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx76 = getelementptr inbounds [11 x ptr], ptr %m_shapes75, i32 0, i32 8
  store ptr %call68, ptr %arrayidx76, align 4
  %call77 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %19 = load float, ptr %scale.addr, align 4
  %mul78 = fmul float 0x3FA99999A0000000, %19
  %20 = load float, ptr %scale.addr, align 4
  %mul79 = fmul float 0x3FD51EB860000000, %20
  %call82 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call77, float %mul78, float %mul79)
          to label %invoke.cont81 unwind label %lpad80

invoke.cont81:                                    ; preds = %invoke.cont72
  %m_shapes84 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx85 = getelementptr inbounds [11 x ptr], ptr %m_shapes84, i32 0, i32 9
  store ptr %call77, ptr %arrayidx85, align 4
  %call86 = call ptr @_ZN13btConvexShapenwEm(i32 56)
  %21 = load float, ptr %scale.addr, align 4
  %mul87 = fmul float 0x3FA47AE140000000, %21
  %22 = load float, ptr %scale.addr, align 4
  %mul88 = fmul float 2.500000e-01, %22
  %call91 = invoke ptr @_ZN14btCapsuleShapeC1Eff(ptr %call86, float %mul87, float %mul88)
          to label %invoke.cont90 unwind label %lpad89

invoke.cont90:                                    ; preds = %invoke.cont81
  %m_shapes93 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx94 = getelementptr inbounds [11 x ptr], ptr %m_shapes93, i32 0, i32 10
  store ptr %call86, ptr %arrayidx94, align 4
  %call95 = call ptr @_ZN11btTransformC1Ev(ptr %offset)
  call void @_ZN11btTransform11setIdentityEv(ptr %offset)
  %23 = load ptr, ptr %positionOffset.addr, align 4
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %offset, ptr %23)
  %call96 = call ptr @_ZN11btTransformC1Ev(ptr %transform)
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0.000000e+00, ptr %ref.tmp98, align 4
  store float 1.000000e+00, ptr %ref.tmp99, align 4
  store float 0.000000e+00, ptr %ref.tmp100, align 4
  %call101 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp97, ptr %ref.tmp98, ptr %ref.tmp99, ptr %ref.tmp100)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp, ptr %scale.addr, ptr %ref.tmp97)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp102, ptr %offset, ptr %transform)
  %m_shapes103 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %24 = load ptr, ptr %m_shapes103, align 4
  %call105 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp102, ptr %24)
  %m_bodies = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  store ptr %call105, ptr %m_bodies, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0.000000e+00, ptr %ref.tmp109, align 4
  store float 0x3FF3333340000000, ptr %ref.tmp110, align 4
  store float 0.000000e+00, ptr %ref.tmp111, align 4
  %call112 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp108, ptr %ref.tmp109, ptr %ref.tmp110, ptr %ref.tmp111)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp107, ptr %scale.addr, ptr %ref.tmp108)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp107)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp113, ptr %offset, ptr %transform)
  %m_shapes114 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx115 = getelementptr inbounds [11 x ptr], ptr %m_shapes114, i32 0, i32 1
  %25 = load ptr, ptr %arrayidx115, align 4
  %call116 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp113, ptr %25)
  %m_bodies117 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx118 = getelementptr inbounds [11 x ptr], ptr %m_bodies117, i32 0, i32 1
  store ptr %call116, ptr %arrayidx118, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0.000000e+00, ptr %ref.tmp121, align 4
  store float 0x3FF99999A0000000, ptr %ref.tmp122, align 4
  store float 0.000000e+00, ptr %ref.tmp123, align 4
  %call124 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp120, ptr %ref.tmp121, ptr %ref.tmp122, ptr %ref.tmp123)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp119, ptr %scale.addr, ptr %ref.tmp120)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp119)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp125, ptr %offset, ptr %transform)
  %m_shapes126 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx127 = getelementptr inbounds [11 x ptr], ptr %m_shapes126, i32 0, i32 2
  %26 = load ptr, ptr %arrayidx127, align 4
  %call128 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp125, ptr %26)
  %m_bodies129 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx130 = getelementptr inbounds [11 x ptr], ptr %m_bodies129, i32 0, i32 2
  store ptr %call128, ptr %arrayidx130, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0xBFC70A3D80000000, ptr %ref.tmp133, align 4
  store float 0x3FE4CCCCC0000000, ptr %ref.tmp134, align 4
  store float 0.000000e+00, ptr %ref.tmp135, align 4
  %call136 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp132, ptr %ref.tmp133, ptr %ref.tmp134, ptr %ref.tmp135)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp131, ptr %scale.addr, ptr %ref.tmp132)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp131)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp137, ptr %offset, ptr %transform)
  %m_shapes138 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx139 = getelementptr inbounds [11 x ptr], ptr %m_shapes138, i32 0, i32 3
  %27 = load ptr, ptr %arrayidx139, align 4
  %call140 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp137, ptr %27)
  %m_bodies141 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx142 = getelementptr inbounds [11 x ptr], ptr %m_bodies141, i32 0, i32 3
  store ptr %call140, ptr %arrayidx142, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0xBFC70A3D80000000, ptr %ref.tmp145, align 4
  store float 0x3FC99999A0000000, ptr %ref.tmp146, align 4
  store float 0.000000e+00, ptr %ref.tmp147, align 4
  %call148 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp144, ptr %ref.tmp145, ptr %ref.tmp146, ptr %ref.tmp147)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp143, ptr %scale.addr, ptr %ref.tmp144)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp143)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp149, ptr %offset, ptr %transform)
  %m_shapes150 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx151 = getelementptr inbounds [11 x ptr], ptr %m_shapes150, i32 0, i32 4
  %28 = load ptr, ptr %arrayidx151, align 4
  %call152 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp149, ptr %28)
  %m_bodies153 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx154 = getelementptr inbounds [11 x ptr], ptr %m_bodies153, i32 0, i32 4
  store ptr %call152, ptr %arrayidx154, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0x3FC70A3D80000000, ptr %ref.tmp157, align 4
  store float 0x3FE4CCCCC0000000, ptr %ref.tmp158, align 4
  store float 0.000000e+00, ptr %ref.tmp159, align 4
  %call160 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp156, ptr %ref.tmp157, ptr %ref.tmp158, ptr %ref.tmp159)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp155, ptr %scale.addr, ptr %ref.tmp156)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp155)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp161, ptr %offset, ptr %transform)
  %m_shapes162 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx163 = getelementptr inbounds [11 x ptr], ptr %m_shapes162, i32 0, i32 5
  %29 = load ptr, ptr %arrayidx163, align 4
  %call164 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp161, ptr %29)
  %m_bodies165 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx166 = getelementptr inbounds [11 x ptr], ptr %m_bodies165, i32 0, i32 5
  store ptr %call164, ptr %arrayidx166, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0x3FC70A3D80000000, ptr %ref.tmp169, align 4
  store float 0x3FC99999A0000000, ptr %ref.tmp170, align 4
  store float 0.000000e+00, ptr %ref.tmp171, align 4
  %call172 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp168, ptr %ref.tmp169, ptr %ref.tmp170, ptr %ref.tmp171)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp167, ptr %scale.addr, ptr %ref.tmp168)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp167)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp173, ptr %offset, ptr %transform)
  %m_shapes174 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx175 = getelementptr inbounds [11 x ptr], ptr %m_shapes174, i32 0, i32 6
  %30 = load ptr, ptr %arrayidx175, align 4
  %call176 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp173, ptr %30)
  %m_bodies177 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx178 = getelementptr inbounds [11 x ptr], ptr %m_bodies177, i32 0, i32 6
  store ptr %call176, ptr %arrayidx178, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0xBFD6666660000000, ptr %ref.tmp181, align 4
  store float 0x3FF7333340000000, ptr %ref.tmp182, align 4
  store float 0.000000e+00, ptr %ref.tmp183, align 4
  %call184 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp180, ptr %ref.tmp181, ptr %ref.tmp182, ptr %ref.tmp183)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp179, ptr %scale.addr, ptr %ref.tmp180)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp179)
  %call185 = call ptr @_ZN11btTransform8getBasisEv(ptr %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call185, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp186, ptr %offset, ptr %transform)
  %m_shapes187 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx188 = getelementptr inbounds [11 x ptr], ptr %m_shapes187, i32 0, i32 7
  %31 = load ptr, ptr %arrayidx188, align 4
  %call189 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp186, ptr %31)
  %m_bodies190 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx191 = getelementptr inbounds [11 x ptr], ptr %m_bodies190, i32 0, i32 7
  store ptr %call189, ptr %arrayidx191, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0xBFE6666660000000, ptr %ref.tmp194, align 4
  store float 0x3FF7333340000000, ptr %ref.tmp195, align 4
  store float 0.000000e+00, ptr %ref.tmp196, align 4
  %call197 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp193, ptr %ref.tmp194, ptr %ref.tmp195, ptr %ref.tmp196)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp192, ptr %scale.addr, ptr %ref.tmp193)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp192)
  %call198 = call ptr @_ZN11btTransform8getBasisEv(ptr %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call198, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp199, ptr %offset, ptr %transform)
  %m_shapes200 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx201 = getelementptr inbounds [11 x ptr], ptr %m_shapes200, i32 0, i32 8
  %32 = load ptr, ptr %arrayidx201, align 4
  %call202 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp199, ptr %32)
  %m_bodies203 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx204 = getelementptr inbounds [11 x ptr], ptr %m_bodies203, i32 0, i32 8
  store ptr %call202, ptr %arrayidx204, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0x3FD6666660000000, ptr %ref.tmp207, align 4
  store float 0x3FF7333340000000, ptr %ref.tmp208, align 4
  store float 0.000000e+00, ptr %ref.tmp209, align 4
  %call210 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp206, ptr %ref.tmp207, ptr %ref.tmp208, ptr %ref.tmp209)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp205, ptr %scale.addr, ptr %ref.tmp206)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp205)
  %call211 = call ptr @_ZN11btTransform8getBasisEv(ptr %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call211, float 0.000000e+00, float 0.000000e+00, float 0xBFF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp212, ptr %offset, ptr %transform)
  %m_shapes213 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx214 = getelementptr inbounds [11 x ptr], ptr %m_shapes213, i32 0, i32 9
  %33 = load ptr, ptr %arrayidx214, align 4
  %call215 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp212, ptr %33)
  %m_bodies216 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx217 = getelementptr inbounds [11 x ptr], ptr %m_bodies216, i32 0, i32 9
  store ptr %call215, ptr %arrayidx217, align 4
  call void @_ZN11btTransform11setIdentityEv(ptr %transform)
  store float 0x3FE6666660000000, ptr %ref.tmp220, align 4
  store float 0x3FF7333340000000, ptr %ref.tmp221, align 4
  store float 0.000000e+00, ptr %ref.tmp222, align 4
  %call223 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp219, ptr %ref.tmp220, ptr %ref.tmp221, ptr %ref.tmp222)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp218, ptr %scale.addr, ptr %ref.tmp219)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %transform, ptr %ref.tmp218)
  %call224 = call ptr @_ZN11btTransform8getBasisEv(ptr %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call224, float 0.000000e+00, float 0.000000e+00, float 0xBFF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(ptr sret(%class.btTransform) %ref.tmp225, ptr %offset, ptr %transform)
  %m_shapes226 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 2
  %arrayidx227 = getelementptr inbounds [11 x ptr], ptr %m_shapes226, i32 0, i32 10
  %34 = load ptr, ptr %arrayidx227, align 4
  %call228 = call ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr %this1, float 1.000000e+00, ptr %ref.tmp225, ptr %34)
  %m_bodies229 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx230 = getelementptr inbounds [11 x ptr], ptr %m_bodies229, i32 0, i32 10
  store ptr %call228, ptr %arrayidx230, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %invoke.cont90
  %35 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %35, 11
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %36 = load i32, ptr %i, align 4
  %m_bodies231 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx232 = getelementptr inbounds [11 x ptr], ptr %m_bodies231, i32 0, i32 %36
  %37 = load ptr, ptr %arrayidx232, align 4
  call void @_ZN11btRigidBody10setDampingEff(ptr %37, float 0x3FA99999A0000000, float 0x3FEB333340000000)
  %38 = load i32, ptr %i, align 4
  %m_bodies233 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx234 = getelementptr inbounds [11 x ptr], ptr %m_bodies233, i32 0, i32 %38
  %39 = load ptr, ptr %arrayidx234, align 4
  call void @_ZN17btCollisionObject19setDeactivationTimeEf(ptr %39, float 0x3FE99999A0000000)
  %40 = load i32, ptr %i, align 4
  %m_bodies235 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx236 = getelementptr inbounds [11 x ptr], ptr %m_bodies235, i32 0, i32 %40
  %41 = load ptr, ptr %arrayidx236, align 4
  call void @_ZN11btRigidBody21setSleepingThresholdsEff(ptr %41, float 0x3FF99999A0000000, float 2.500000e+00)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %42 = load i32, ptr %i, align 4
  %inc = add nsw i32 %42, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

lpad:                                             ; preds = %entry
  %43 = landingpad { ptr, i32 }
          cleanup
  %44 = extractvalue { ptr, i32 } %43, 0
  store ptr %44, ptr %exn.slot
  %45 = extractvalue { ptr, i32 } %43, 1
  store i32 %45, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call)
          to label %invoke.cont4 unwind label %terminate.lpad

invoke.cont4:                                     ; preds = %lpad
  br label %eh.resume

lpad8:                                            ; preds = %invoke.cont
  %46 = landingpad { ptr, i32 }
          cleanup
  %47 = extractvalue { ptr, i32 } %46, 0
  store ptr %47, ptr %exn.slot
  %48 = extractvalue { ptr, i32 } %46, 1
  store i32 %48, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call5)
          to label %invoke.cont11 unwind label %terminate.lpad

invoke.cont11:                                    ; preds = %lpad8
  br label %eh.resume

lpad17:                                           ; preds = %invoke.cont9
  %49 = landingpad { ptr, i32 }
          cleanup
  %50 = extractvalue { ptr, i32 } %49, 0
  store ptr %50, ptr %exn.slot
  %51 = extractvalue { ptr, i32 } %49, 1
  store i32 %51, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call14)
          to label %invoke.cont20 unwind label %terminate.lpad

invoke.cont20:                                    ; preds = %lpad17
  br label %eh.resume

lpad26:                                           ; preds = %invoke.cont18
  %52 = landingpad { ptr, i32 }
          cleanup
  %53 = extractvalue { ptr, i32 } %52, 0
  store ptr %53, ptr %exn.slot
  %54 = extractvalue { ptr, i32 } %52, 1
  store i32 %54, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call23)
          to label %invoke.cont29 unwind label %terminate.lpad

invoke.cont29:                                    ; preds = %lpad26
  br label %eh.resume

lpad35:                                           ; preds = %invoke.cont27
  %55 = landingpad { ptr, i32 }
          cleanup
  %56 = extractvalue { ptr, i32 } %55, 0
  store ptr %56, ptr %exn.slot
  %57 = extractvalue { ptr, i32 } %55, 1
  store i32 %57, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call32)
          to label %invoke.cont38 unwind label %terminate.lpad

invoke.cont38:                                    ; preds = %lpad35
  br label %eh.resume

lpad44:                                           ; preds = %invoke.cont36
  %58 = landingpad { ptr, i32 }
          cleanup
  %59 = extractvalue { ptr, i32 } %58, 0
  store ptr %59, ptr %exn.slot
  %60 = extractvalue { ptr, i32 } %58, 1
  store i32 %60, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call41)
          to label %invoke.cont47 unwind label %terminate.lpad

invoke.cont47:                                    ; preds = %lpad44
  br label %eh.resume

lpad53:                                           ; preds = %invoke.cont45
  %61 = landingpad { ptr, i32 }
          cleanup
  %62 = extractvalue { ptr, i32 } %61, 0
  store ptr %62, ptr %exn.slot
  %63 = extractvalue { ptr, i32 } %61, 1
  store i32 %63, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call50)
          to label %invoke.cont56 unwind label %terminate.lpad

invoke.cont56:                                    ; preds = %lpad53
  br label %eh.resume

lpad62:                                           ; preds = %invoke.cont54
  %64 = landingpad { ptr, i32 }
          cleanup
  %65 = extractvalue { ptr, i32 } %64, 0
  store ptr %65, ptr %exn.slot
  %66 = extractvalue { ptr, i32 } %64, 1
  store i32 %66, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call59)
          to label %invoke.cont65 unwind label %terminate.lpad

invoke.cont65:                                    ; preds = %lpad62
  br label %eh.resume

lpad71:                                           ; preds = %invoke.cont63
  %67 = landingpad { ptr, i32 }
          cleanup
  %68 = extractvalue { ptr, i32 } %67, 0
  store ptr %68, ptr %exn.slot
  %69 = extractvalue { ptr, i32 } %67, 1
  store i32 %69, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call68)
          to label %invoke.cont74 unwind label %terminate.lpad

invoke.cont74:                                    ; preds = %lpad71
  br label %eh.resume

lpad80:                                           ; preds = %invoke.cont72
  %70 = landingpad { ptr, i32 }
          cleanup
  %71 = extractvalue { ptr, i32 } %70, 0
  store ptr %71, ptr %exn.slot
  %72 = extractvalue { ptr, i32 } %70, 1
  store i32 %72, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call77)
          to label %invoke.cont83 unwind label %terminate.lpad

invoke.cont83:                                    ; preds = %lpad80
  br label %eh.resume

lpad89:                                           ; preds = %invoke.cont81
  %73 = landingpad { ptr, i32 }
          cleanup
  %74 = extractvalue { ptr, i32 } %73, 0
  store ptr %74, ptr %exn.slot
  %75 = extractvalue { ptr, i32 } %73, 1
  store i32 %75, ptr %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(ptr %call86)
          to label %invoke.cont92 unwind label %terminate.lpad

invoke.cont92:                                    ; preds = %lpad89
  br label %eh.resume

for.end:                                          ; preds = %for.cond
  %call237 = call ptr @_ZN11btTransformC1Ev(ptr %localA)
  %call238 = call ptr @_ZN11btTransformC1Ev(ptr %localB)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call239 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call239, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp242, align 4
  store float 0x3FC3333340000000, ptr %ref.tmp243, align 4
  store float 0.000000e+00, ptr %ref.tmp244, align 4
  %call245 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp241, ptr %ref.tmp242, ptr %ref.tmp243, ptr %ref.tmp244)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp240, ptr %scale.addr, ptr %ref.tmp241)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp240)
  %call246 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call246, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp249, align 4
  store float 0xBFC3333340000000, ptr %ref.tmp250, align 4
  store float 0.000000e+00, ptr %ref.tmp251, align 4
  %call252 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp248, ptr %ref.tmp249, ptr %ref.tmp250, ptr %ref.tmp251)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp247, ptr %scale.addr, ptr %ref.tmp248)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp247)
  %call253 = call noalias ptr @_Znwm(i32 780)
  %m_bodies254 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %76 = load ptr, ptr %m_bodies254, align 4
  %m_bodies256 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx257 = getelementptr inbounds [11 x ptr], ptr %m_bodies256, i32 0, i32 1
  %77 = load ptr, ptr %arrayidx257, align 4
  %call260 = invoke ptr @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(ptr %call253, ptr %76, ptr %77, ptr %localA, ptr %localB, i1 zeroext false)
          to label %invoke.cont259 unwind label %lpad258

invoke.cont259:                                   ; preds = %for.end
  store ptr %call253, ptr %hingeC, align 4
  %78 = load ptr, ptr %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(ptr %78, float 0xBFE921FB60000000, float 0x3FF921FB60000000, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %79 = load ptr, ptr %hingeC, align 4
  %m_joints = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  store ptr %79, ptr %m_joints, align 4
  %m_ownerWorld262 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %80 = load ptr, ptr %m_ownerWorld262, align 4
  %vtable = load ptr, ptr %80
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 10
  %81 = load ptr, ptr %vfn
  %m_joints263 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %82 = load ptr, ptr %m_joints263, align 4
  call void %81(ptr %80, ptr %82, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call265 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call265, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, ptr %ref.tmp268, align 4
  store float 0x3FD3333340000000, ptr %ref.tmp269, align 4
  store float 0.000000e+00, ptr %ref.tmp270, align 4
  %call271 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp267, ptr %ref.tmp268, ptr %ref.tmp269, ptr %ref.tmp270)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp266, ptr %scale.addr, ptr %ref.tmp267)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp266)
  %call272 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call272, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, ptr %ref.tmp275, align 4
  store float 0xBFC1EB8520000000, ptr %ref.tmp276, align 4
  store float 0.000000e+00, ptr %ref.tmp277, align 4
  %call278 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp274, ptr %ref.tmp275, ptr %ref.tmp276, ptr %ref.tmp277)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp273, ptr %scale.addr, ptr %ref.tmp274)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp273)
  %call279 = call noalias ptr @_Znwm(i32 628)
  %m_bodies280 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx281 = getelementptr inbounds [11 x ptr], ptr %m_bodies280, i32 0, i32 1
  %83 = load ptr, ptr %arrayidx281, align 4
  %m_bodies282 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx283 = getelementptr inbounds [11 x ptr], ptr %m_bodies282, i32 0, i32 2
  %84 = load ptr, ptr %arrayidx283, align 4
  %call286 = invoke ptr @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(ptr %call279, ptr %83, ptr %84, ptr %localA, ptr %localB)
          to label %invoke.cont285 unwind label %lpad284

invoke.cont285:                                   ; preds = %invoke.cont259
  store ptr %call279, ptr %coneC, align 4
  %85 = load ptr, ptr %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(ptr %85, float 0x3FE921FB60000000, float 0x3FE921FB60000000, float 0x3FF921FB60000000, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %86 = load ptr, ptr %coneC, align 4
  %m_joints287 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx288 = getelementptr inbounds [10 x ptr], ptr %m_joints287, i32 0, i32 1
  store ptr %86, ptr %arrayidx288, align 4
  %m_ownerWorld289 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %87 = load ptr, ptr %m_ownerWorld289, align 4
  %vtable290 = load ptr, ptr %87
  %vfn291 = getelementptr inbounds ptr, ptr %vtable290, i64 10
  %88 = load ptr, ptr %vfn291
  %m_joints292 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx293 = getelementptr inbounds [10 x ptr], ptr %m_joints292, i32 0, i32 1
  %89 = load ptr, ptr %arrayidx293, align 4
  call void %88(ptr %87, ptr %89, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call294 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call294, float 0.000000e+00, float 0.000000e+00, float 0xC00F6A7A20000000)
  store float 0xBFC70A3D80000000, ptr %ref.tmp297, align 4
  store float 0xBFB99999A0000000, ptr %ref.tmp298, align 4
  store float 0.000000e+00, ptr %ref.tmp299, align 4
  %call300 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp296, ptr %ref.tmp297, ptr %ref.tmp298, ptr %ref.tmp299)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp295, ptr %scale.addr, ptr %ref.tmp296)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp295)
  %call301 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call301, float 0.000000e+00, float 0.000000e+00, float 0xC00F6A7A20000000)
  store float 0.000000e+00, ptr %ref.tmp304, align 4
  store float 0x3FCCCCCCC0000000, ptr %ref.tmp305, align 4
  store float 0.000000e+00, ptr %ref.tmp306, align 4
  %call307 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp303, ptr %ref.tmp304, ptr %ref.tmp305, ptr %ref.tmp306)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp302, ptr %scale.addr, ptr %ref.tmp303)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp302)
  %call308 = call noalias ptr @_Znwm(i32 628)
  %m_bodies309 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %90 = load ptr, ptr %m_bodies309, align 4
  %m_bodies311 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx312 = getelementptr inbounds [11 x ptr], ptr %m_bodies311, i32 0, i32 3
  %91 = load ptr, ptr %arrayidx312, align 4
  %call315 = invoke ptr @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(ptr %call308, ptr %90, ptr %91, ptr %localA, ptr %localB)
          to label %invoke.cont314 unwind label %lpad313

invoke.cont314:                                   ; preds = %invoke.cont285
  store ptr %call308, ptr %coneC, align 4
  %92 = load ptr, ptr %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(ptr %92, float 0x3FE921FB60000000, float 0x3FE921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %93 = load ptr, ptr %coneC, align 4
  %m_joints316 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx317 = getelementptr inbounds [10 x ptr], ptr %m_joints316, i32 0, i32 2
  store ptr %93, ptr %arrayidx317, align 4
  %m_ownerWorld318 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %94 = load ptr, ptr %m_ownerWorld318, align 4
  %vtable319 = load ptr, ptr %94
  %vfn320 = getelementptr inbounds ptr, ptr %vtable319, i64 10
  %95 = load ptr, ptr %vfn320
  %m_joints321 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx322 = getelementptr inbounds [10 x ptr], ptr %m_joints321, i32 0, i32 2
  %96 = load ptr, ptr %arrayidx322, align 4
  call void %95(ptr %94, ptr %96, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call323 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call323, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp326, align 4
  store float 0xBFCCCCCCC0000000, ptr %ref.tmp327, align 4
  store float 0.000000e+00, ptr %ref.tmp328, align 4
  %call329 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp325, ptr %ref.tmp326, ptr %ref.tmp327, ptr %ref.tmp328)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp324, ptr %scale.addr, ptr %ref.tmp325)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp324)
  %call330 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call330, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp333, align 4
  store float 0x3FC7AE1480000000, ptr %ref.tmp334, align 4
  store float 0.000000e+00, ptr %ref.tmp335, align 4
  %call336 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp332, ptr %ref.tmp333, ptr %ref.tmp334, ptr %ref.tmp335)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp331, ptr %scale.addr, ptr %ref.tmp332)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp331)
  %call337 = call noalias ptr @_Znwm(i32 780)
  %m_bodies338 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx339 = getelementptr inbounds [11 x ptr], ptr %m_bodies338, i32 0, i32 3
  %97 = load ptr, ptr %arrayidx339, align 4
  %m_bodies340 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx341 = getelementptr inbounds [11 x ptr], ptr %m_bodies340, i32 0, i32 4
  %98 = load ptr, ptr %arrayidx341, align 4
  %call344 = invoke ptr @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(ptr %call337, ptr %97, ptr %98, ptr %localA, ptr %localB, i1 zeroext false)
          to label %invoke.cont343 unwind label %lpad342

invoke.cont343:                                   ; preds = %invoke.cont314
  store ptr %call337, ptr %hingeC, align 4
  %99 = load ptr, ptr %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(ptr %99, float 0.000000e+00, float 0x3FF921FB60000000, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %100 = load ptr, ptr %hingeC, align 4
  %m_joints345 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx346 = getelementptr inbounds [10 x ptr], ptr %m_joints345, i32 0, i32 3
  store ptr %100, ptr %arrayidx346, align 4
  %m_ownerWorld347 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %101 = load ptr, ptr %m_ownerWorld347, align 4
  %vtable348 = load ptr, ptr %101
  %vfn349 = getelementptr inbounds ptr, ptr %vtable348, i64 10
  %102 = load ptr, ptr %vfn349
  %m_joints350 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx351 = getelementptr inbounds [10 x ptr], ptr %m_joints350, i32 0, i32 3
  %103 = load ptr, ptr %arrayidx351, align 4
  call void %102(ptr %101, ptr %103, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call352 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call352, float 0.000000e+00, float 0.000000e+00, float 0x3FE921FB60000000)
  store float 0x3FC70A3D80000000, ptr %ref.tmp355, align 4
  store float 0xBFB99999A0000000, ptr %ref.tmp356, align 4
  store float 0.000000e+00, ptr %ref.tmp357, align 4
  %call358 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp354, ptr %ref.tmp355, ptr %ref.tmp356, ptr %ref.tmp357)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp353, ptr %scale.addr, ptr %ref.tmp354)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp353)
  %call359 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call359, float 0.000000e+00, float 0.000000e+00, float 0x3FE921FB60000000)
  store float 0.000000e+00, ptr %ref.tmp362, align 4
  store float 0x3FCCCCCCC0000000, ptr %ref.tmp363, align 4
  store float 0.000000e+00, ptr %ref.tmp364, align 4
  %call365 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp361, ptr %ref.tmp362, ptr %ref.tmp363, ptr %ref.tmp364)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp360, ptr %scale.addr, ptr %ref.tmp361)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp360)
  %call366 = call noalias ptr @_Znwm(i32 628)
  %m_bodies367 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %104 = load ptr, ptr %m_bodies367, align 4
  %m_bodies369 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx370 = getelementptr inbounds [11 x ptr], ptr %m_bodies369, i32 0, i32 5
  %105 = load ptr, ptr %arrayidx370, align 4
  %call373 = invoke ptr @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(ptr %call366, ptr %104, ptr %105, ptr %localA, ptr %localB)
          to label %invoke.cont372 unwind label %lpad371

invoke.cont372:                                   ; preds = %invoke.cont343
  store ptr %call366, ptr %coneC, align 4
  %106 = load ptr, ptr %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(ptr %106, float 0x3FE921FB60000000, float 0x3FE921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %107 = load ptr, ptr %coneC, align 4
  %m_joints374 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx375 = getelementptr inbounds [10 x ptr], ptr %m_joints374, i32 0, i32 4
  store ptr %107, ptr %arrayidx375, align 4
  %m_ownerWorld376 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %108 = load ptr, ptr %m_ownerWorld376, align 4
  %vtable377 = load ptr, ptr %108
  %vfn378 = getelementptr inbounds ptr, ptr %vtable377, i64 10
  %109 = load ptr, ptr %vfn378
  %m_joints379 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx380 = getelementptr inbounds [10 x ptr], ptr %m_joints379, i32 0, i32 4
  %110 = load ptr, ptr %arrayidx380, align 4
  call void %109(ptr %108, ptr %110, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call381 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call381, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp384, align 4
  store float 0xBFCCCCCCC0000000, ptr %ref.tmp385, align 4
  store float 0.000000e+00, ptr %ref.tmp386, align 4
  %call387 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp383, ptr %ref.tmp384, ptr %ref.tmp385, ptr %ref.tmp386)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp382, ptr %scale.addr, ptr %ref.tmp383)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp382)
  %call388 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call388, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp391, align 4
  store float 0x3FC7AE1480000000, ptr %ref.tmp392, align 4
  store float 0.000000e+00, ptr %ref.tmp393, align 4
  %call394 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp390, ptr %ref.tmp391, ptr %ref.tmp392, ptr %ref.tmp393)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp389, ptr %scale.addr, ptr %ref.tmp390)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp389)
  %call395 = call noalias ptr @_Znwm(i32 780)
  %m_bodies396 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx397 = getelementptr inbounds [11 x ptr], ptr %m_bodies396, i32 0, i32 5
  %111 = load ptr, ptr %arrayidx397, align 4
  %m_bodies398 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx399 = getelementptr inbounds [11 x ptr], ptr %m_bodies398, i32 0, i32 6
  %112 = load ptr, ptr %arrayidx399, align 4
  %call402 = invoke ptr @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(ptr %call395, ptr %111, ptr %112, ptr %localA, ptr %localB, i1 zeroext false)
          to label %invoke.cont401 unwind label %lpad400

invoke.cont401:                                   ; preds = %invoke.cont372
  store ptr %call395, ptr %hingeC, align 4
  %113 = load ptr, ptr %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(ptr %113, float 0.000000e+00, float 0x3FF921FB60000000, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %114 = load ptr, ptr %hingeC, align 4
  %m_joints403 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx404 = getelementptr inbounds [10 x ptr], ptr %m_joints403, i32 0, i32 5
  store ptr %114, ptr %arrayidx404, align 4
  %m_ownerWorld405 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %115 = load ptr, ptr %m_ownerWorld405, align 4
  %vtable406 = load ptr, ptr %115
  %vfn407 = getelementptr inbounds ptr, ptr %vtable406, i64 10
  %116 = load ptr, ptr %vfn407
  %m_joints408 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx409 = getelementptr inbounds [10 x ptr], ptr %m_joints408, i32 0, i32 5
  %117 = load ptr, ptr %arrayidx409, align 4
  call void %116(ptr %115, ptr %117, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call410 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call410, float 0.000000e+00, float 0.000000e+00, float 0x400921FB60000000)
  store float 0xBFC99999A0000000, ptr %ref.tmp413, align 4
  store float 0x3FC3333340000000, ptr %ref.tmp414, align 4
  store float 0.000000e+00, ptr %ref.tmp415, align 4
  %call416 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp412, ptr %ref.tmp413, ptr %ref.tmp414, ptr %ref.tmp415)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp411, ptr %scale.addr, ptr %ref.tmp412)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp411)
  %call417 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call417, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, ptr %ref.tmp420, align 4
  store float 0xBFC70A3D80000000, ptr %ref.tmp421, align 4
  store float 0.000000e+00, ptr %ref.tmp422, align 4
  %call423 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp419, ptr %ref.tmp420, ptr %ref.tmp421, ptr %ref.tmp422)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp418, ptr %scale.addr, ptr %ref.tmp419)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp418)
  %call424 = call noalias ptr @_Znwm(i32 628)
  %m_bodies425 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx426 = getelementptr inbounds [11 x ptr], ptr %m_bodies425, i32 0, i32 1
  %118 = load ptr, ptr %arrayidx426, align 4
  %m_bodies427 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx428 = getelementptr inbounds [11 x ptr], ptr %m_bodies427, i32 0, i32 7
  %119 = load ptr, ptr %arrayidx428, align 4
  %call431 = invoke ptr @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(ptr %call424, ptr %118, ptr %119, ptr %localA, ptr %localB)
          to label %invoke.cont430 unwind label %lpad429

invoke.cont430:                                   ; preds = %invoke.cont401
  store ptr %call424, ptr %coneC, align 4
  %120 = load ptr, ptr %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(ptr %120, float 0x3FF921FB60000000, float 0x3FF921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %121 = load ptr, ptr %coneC, align 4
  %m_joints432 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx433 = getelementptr inbounds [10 x ptr], ptr %m_joints432, i32 0, i32 6
  store ptr %121, ptr %arrayidx433, align 4
  %m_ownerWorld434 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %122 = load ptr, ptr %m_ownerWorld434, align 4
  %vtable435 = load ptr, ptr %122
  %vfn436 = getelementptr inbounds ptr, ptr %vtable435, i64 10
  %123 = load ptr, ptr %vfn436
  %m_joints437 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx438 = getelementptr inbounds [10 x ptr], ptr %m_joints437, i32 0, i32 6
  %124 = load ptr, ptr %arrayidx438, align 4
  call void %123(ptr %122, ptr %124, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call439 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call439, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp442, align 4
  store float 0x3FC70A3D80000000, ptr %ref.tmp443, align 4
  store float 0.000000e+00, ptr %ref.tmp444, align 4
  %call445 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp441, ptr %ref.tmp442, ptr %ref.tmp443, ptr %ref.tmp444)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp440, ptr %scale.addr, ptr %ref.tmp441)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp440)
  %call446 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call446, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp449, align 4
  store float 0xBFC1EB8520000000, ptr %ref.tmp450, align 4
  store float 0.000000e+00, ptr %ref.tmp451, align 4
  %call452 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp448, ptr %ref.tmp449, ptr %ref.tmp450, ptr %ref.tmp451)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp447, ptr %scale.addr, ptr %ref.tmp448)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp447)
  %call453 = call noalias ptr @_Znwm(i32 780)
  %m_bodies454 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx455 = getelementptr inbounds [11 x ptr], ptr %m_bodies454, i32 0, i32 7
  %125 = load ptr, ptr %arrayidx455, align 4
  %m_bodies456 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx457 = getelementptr inbounds [11 x ptr], ptr %m_bodies456, i32 0, i32 8
  %126 = load ptr, ptr %arrayidx457, align 4
  %call460 = invoke ptr @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(ptr %call453, ptr %125, ptr %126, ptr %localA, ptr %localB, i1 zeroext false)
          to label %invoke.cont459 unwind label %lpad458

invoke.cont459:                                   ; preds = %invoke.cont430
  store ptr %call453, ptr %hingeC, align 4
  %127 = load ptr, ptr %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(ptr %127, float 0xBFF921FB60000000, float 0.000000e+00, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %128 = load ptr, ptr %hingeC, align 4
  %m_joints461 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx462 = getelementptr inbounds [10 x ptr], ptr %m_joints461, i32 0, i32 7
  store ptr %128, ptr %arrayidx462, align 4
  %m_ownerWorld463 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %129 = load ptr, ptr %m_ownerWorld463, align 4
  %vtable464 = load ptr, ptr %129
  %vfn465 = getelementptr inbounds ptr, ptr %vtable464, i64 10
  %130 = load ptr, ptr %vfn465
  %m_joints466 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx467 = getelementptr inbounds [10 x ptr], ptr %m_joints466, i32 0, i32 7
  %131 = load ptr, ptr %arrayidx467, align 4
  call void %130(ptr %129, ptr %131, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call468 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call468, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
  store float 0x3FC99999A0000000, ptr %ref.tmp471, align 4
  store float 0x3FC3333340000000, ptr %ref.tmp472, align 4
  store float 0.000000e+00, ptr %ref.tmp473, align 4
  %call474 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp470, ptr %ref.tmp471, ptr %ref.tmp472, ptr %ref.tmp473)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp469, ptr %scale.addr, ptr %ref.tmp470)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp469)
  %call475 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call475, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, ptr %ref.tmp478, align 4
  store float 0xBFC70A3D80000000, ptr %ref.tmp479, align 4
  store float 0.000000e+00, ptr %ref.tmp480, align 4
  %call481 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp477, ptr %ref.tmp478, ptr %ref.tmp479, ptr %ref.tmp480)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp476, ptr %scale.addr, ptr %ref.tmp477)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp476)
  %call482 = call noalias ptr @_Znwm(i32 628)
  %m_bodies483 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx484 = getelementptr inbounds [11 x ptr], ptr %m_bodies483, i32 0, i32 1
  %132 = load ptr, ptr %arrayidx484, align 4
  %m_bodies485 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx486 = getelementptr inbounds [11 x ptr], ptr %m_bodies485, i32 0, i32 9
  %133 = load ptr, ptr %arrayidx486, align 4
  %call489 = invoke ptr @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(ptr %call482, ptr %132, ptr %133, ptr %localA, ptr %localB)
          to label %invoke.cont488 unwind label %lpad487

invoke.cont488:                                   ; preds = %invoke.cont459
  store ptr %call482, ptr %coneC, align 4
  %134 = load ptr, ptr %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(ptr %134, float 0x3FF921FB60000000, float 0x3FF921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %135 = load ptr, ptr %coneC, align 4
  %m_joints490 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx491 = getelementptr inbounds [10 x ptr], ptr %m_joints490, i32 0, i32 8
  store ptr %135, ptr %arrayidx491, align 4
  %m_ownerWorld492 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %136 = load ptr, ptr %m_ownerWorld492, align 4
  %vtable493 = load ptr, ptr %136
  %vfn494 = getelementptr inbounds ptr, ptr %vtable493, i64 10
  %137 = load ptr, ptr %vfn494
  %m_joints495 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx496 = getelementptr inbounds [10 x ptr], ptr %m_joints495, i32 0, i32 8
  %138 = load ptr, ptr %arrayidx496, align 4
  call void %137(ptr %136, ptr %138, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(ptr %localA)
  call void @_ZN11btTransform11setIdentityEv(ptr %localB)
  %call497 = call ptr @_ZN11btTransform8getBasisEv(ptr %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call497, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp500, align 4
  store float 0x3FC70A3D80000000, ptr %ref.tmp501, align 4
  store float 0.000000e+00, ptr %ref.tmp502, align 4
  %call503 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp499, ptr %ref.tmp500, ptr %ref.tmp501, ptr %ref.tmp502)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp498, ptr %scale.addr, ptr %ref.tmp499)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localA, ptr %ref.tmp498)
  %call504 = call ptr @_ZN11btTransform8getBasisEv(ptr %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(ptr %call504, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, ptr %ref.tmp507, align 4
  store float 0xBFC1EB8520000000, ptr %ref.tmp508, align 4
  store float 0.000000e+00, ptr %ref.tmp509, align 4
  %call510 = call ptr @_ZN9btVector3C1ERKfS1_S1_(ptr %ref.tmp506, ptr %ref.tmp507, ptr %ref.tmp508, ptr %ref.tmp509)
  call void @_ZmlRKfRK9btVector3(ptr sret(%class.btVector3) %ref.tmp505, ptr %scale.addr, ptr %ref.tmp506)
  call void @_ZN11btTransform9setOriginERK9btVector3(ptr %localB, ptr %ref.tmp505)
  %call511 = call noalias ptr @_Znwm(i32 780)
  %m_bodies512 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx513 = getelementptr inbounds [11 x ptr], ptr %m_bodies512, i32 0, i32 9
  %139 = load ptr, ptr %arrayidx513, align 4
  %m_bodies514 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 3
  %arrayidx515 = getelementptr inbounds [11 x ptr], ptr %m_bodies514, i32 0, i32 10
  %140 = load ptr, ptr %arrayidx515, align 4
  %call518 = invoke ptr @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(ptr %call511, ptr %139, ptr %140, ptr %localA, ptr %localB, i1 zeroext false)
          to label %invoke.cont517 unwind label %lpad516

invoke.cont517:                                   ; preds = %invoke.cont488
  store ptr %call511, ptr %hingeC, align 4
  %141 = load ptr, ptr %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(ptr %141, float 0xBFF921FB60000000, float 0.000000e+00, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %142 = load ptr, ptr %hingeC, align 4
  %m_joints519 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx520 = getelementptr inbounds [10 x ptr], ptr %m_joints519, i32 0, i32 9
  store ptr %142, ptr %arrayidx520, align 4
  %m_ownerWorld521 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 1
  %143 = load ptr, ptr %m_ownerWorld521, align 4
  %vtable522 = load ptr, ptr %143
  %vfn523 = getelementptr inbounds ptr, ptr %vtable522, i64 10
  %144 = load ptr, ptr %vfn523
  %m_joints524 = getelementptr inbounds %class.RagDoll, ptr %this1, i32 0, i32 4
  %arrayidx525 = getelementptr inbounds [10 x ptr], ptr %m_joints524, i32 0, i32 9
  %145 = load ptr, ptr %arrayidx525, align 4
  call void %144(ptr %143, ptr %145, i1 zeroext true)
  %146 = load ptr, ptr %retval
  ret ptr %146

lpad258:                                          ; preds = %for.end
  %147 = landingpad { ptr, i32 }
          cleanup
  %148 = extractvalue { ptr, i32 } %147, 0
  store ptr %148, ptr %exn.slot
  %149 = extractvalue { ptr, i32 } %147, 1
  store i32 %149, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call253) nounwind
  br label %eh.resume

lpad284:                                          ; preds = %invoke.cont259
  %150 = landingpad { ptr, i32 }
          cleanup
  %151 = extractvalue { ptr, i32 } %150, 0
  store ptr %151, ptr %exn.slot
  %152 = extractvalue { ptr, i32 } %150, 1
  store i32 %152, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call279) nounwind
  br label %eh.resume

lpad313:                                          ; preds = %invoke.cont285
  %153 = landingpad { ptr, i32 }
          cleanup
  %154 = extractvalue { ptr, i32 } %153, 0
  store ptr %154, ptr %exn.slot
  %155 = extractvalue { ptr, i32 } %153, 1
  store i32 %155, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call308) nounwind
  br label %eh.resume

lpad342:                                          ; preds = %invoke.cont314
  %156 = landingpad { ptr, i32 }
          cleanup
  %157 = extractvalue { ptr, i32 } %156, 0
  store ptr %157, ptr %exn.slot
  %158 = extractvalue { ptr, i32 } %156, 1
  store i32 %158, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call337) nounwind
  br label %eh.resume

lpad371:                                          ; preds = %invoke.cont343
  %159 = landingpad { ptr, i32 }
          cleanup
  %160 = extractvalue { ptr, i32 } %159, 0
  store ptr %160, ptr %exn.slot
  %161 = extractvalue { ptr, i32 } %159, 1
  store i32 %161, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call366) nounwind
  br label %eh.resume

lpad400:                                          ; preds = %invoke.cont372
  %162 = landingpad { ptr, i32 }
          cleanup
  %163 = extractvalue { ptr, i32 } %162, 0
  store ptr %163, ptr %exn.slot
  %164 = extractvalue { ptr, i32 } %162, 1
  store i32 %164, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call395) nounwind
  br label %eh.resume

lpad429:                                          ; preds = %invoke.cont401
  %165 = landingpad { ptr, i32 }
          cleanup
  %166 = extractvalue { ptr, i32 } %165, 0
  store ptr %166, ptr %exn.slot
  %167 = extractvalue { ptr, i32 } %165, 1
  store i32 %167, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call424) nounwind
  br label %eh.resume

lpad458:                                          ; preds = %invoke.cont430
  %168 = landingpad { ptr, i32 }
          cleanup
  %169 = extractvalue { ptr, i32 } %168, 0
  store ptr %169, ptr %exn.slot
  %170 = extractvalue { ptr, i32 } %168, 1
  store i32 %170, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call453) nounwind
  br label %eh.resume

lpad487:                                          ; preds = %invoke.cont459
  %171 = landingpad { ptr, i32 }
          cleanup
  %172 = extractvalue { ptr, i32 } %171, 0
  store ptr %172, ptr %exn.slot
  %173 = extractvalue { ptr, i32 } %171, 1
  store i32 %173, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call482) nounwind
  br label %eh.resume

lpad516:                                          ; preds = %invoke.cont488
  %174 = landingpad { ptr, i32 }
          cleanup
  %175 = extractvalue { ptr, i32 } %174, 0
  store ptr %175, ptr %exn.slot
  %176 = extractvalue { ptr, i32 } %174, 1
  store i32 %176, ptr %ehselector.slot
  call void @_ZdlPv(ptr %call511) nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad516, %lpad487, %lpad458, %lpad429, %lpad400, %lpad371, %lpad342, %lpad313, %lpad284, %lpad258, %invoke.cont92, %invoke.cont83, %invoke.cont74, %invoke.cont65, %invoke.cont56, %invoke.cont47, %invoke.cont38, %invoke.cont29, %invoke.cont20, %invoke.cont11, %invoke.cont4
  %exn = load ptr, ptr %exn.slot
  %sel = load i32, ptr %ehselector.slot
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn, 0
  %lpad.val526 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
  resume { ptr, i32 } %lpad.val526

terminate.lpad:                                   ; preds = %lpad89, %lpad80, %lpad71, %lpad62, %lpad53, %lpad44, %lpad35, %lpad26, %lpad17, %lpad8, %lpad
  %177 = landingpad { ptr, i32 }
          catch ptr null
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable
}

declare void @_ZmlRKfRK9btVector3(ptr noalias sret(%class.btVector3), ptr, ptr) inlinehint ssp

declare ptr @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(ptr, float, ptr, ptr) ssp align 2

declare void @_ZNK11btTransformmlERKS_(ptr noalias sret(%class.btTransform), ptr, ptr) inlinehint ssp align 2

declare void @_ZN11btMatrix3x311setEulerZYXEfff(ptr, float, float, float) ssp align 2

declare void @_ZN11btRigidBody10setDampingEff(ptr, float, float)

declare void @_ZN17btCollisionObject19setDeactivationTimeEf(ptr, float) nounwind ssp align 2

declare void @_ZN11btRigidBody21setSleepingThresholdsEff(ptr, float, float) nounwind ssp align 2

declare ptr @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(ptr, ptr, ptr, ptr, ptr, i1 zeroext)

declare void @_ZN17btHingeConstraint8setLimitEfffff(ptr, float, float, float, float, float) ssp align 2

declare ptr @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(ptr, ptr, ptr, ptr, ptr)

declare void @_ZN21btConeTwistConstraint8setLimitEffffff(ptr, float, float, float, float, float, float) nounwind ssp align 2
