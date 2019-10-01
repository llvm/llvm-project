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
%class.btCollisionShape = type { i32 (...)**, i32, i8* }
%class.RagDoll = type { i32 (...)**, %class.btDynamicsWorld*, [11 x %class.btCollisionShape*], [11 x %class.btRigidBody*], [10 x %class.btTypedConstraint*] }
%class.btDynamicsWorld = type { %class.btCollisionWorld, void (%class.btDynamicsWorld*, float)*, void (%class.btDynamicsWorld*, float)*, i8*, %struct.btContactSolverInfo }
%class.btCollisionWorld = type { i32 (...)**, %class.btAlignedObjectArray, %class.btDispatcher*, %struct.btDispatcherInfo, %class.btStackAlloc*, %class.btBroadphaseInterface*, %class.btIDebugDraw*, i8 }
%class.btAlignedObjectArray = type { %class.btAlignedAllocator, i32, i32, %class.btCollisionObject**, i8 }
%class.btAlignedAllocator = type { i8 }
%class.btCollisionObject = type { i32 (...)**, %class.btTransform, %class.btTransform, %class.btVector3, %class.btVector3, %class.btVector3, i8, float, %struct.btBroadphaseProxy*, %class.btCollisionShape*, %class.btCollisionShape*, i32, i32, i32, i32, float, float, float, i8*, i32, float, float, float, i8, [7 x i8] }
%struct.btBroadphaseProxy = type { i8*, i16, i16, i8*, i32, %class.btVector3, %class.btVector3 }
%class.btDispatcher = type { i32 (...)** }
%struct.btDispatcherInfo = type { float, i32, i32, float, i8, %class.btIDebugDraw*, i8, i8, i8, float, i8, float, %class.btStackAlloc* }
%class.btIDebugDraw = type { i32 (...)** }
%class.btStackAlloc = type opaque
%class.btBroadphaseInterface = type { i32 (...)** }
%struct.btContactSolverInfo = type { %struct.btContactSolverInfoData }
%struct.btContactSolverInfoData = type { float, float, float, float, float, i32, float, float, float, float, float, i32, float, float, float, i32, i32 }
%class.btRigidBody = type { %class.btCollisionObject, %class.btMatrix3x3, %class.btVector3, %class.btVector3, float, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, float, float, i8, float, float, float, float, float, float, %class.btMotionState*, %class.btAlignedObjectArray.22, i32, i32, i32 }
%class.btMotionState = type { i32 (...)** }
%class.btAlignedObjectArray.22 = type { %class.btAlignedAllocator.23, i32, i32, %class.btTypedConstraint**, i8 }
%class.btAlignedAllocator.23 = type { i8 }
%class.btTypedConstraint = type { i32 (...)**, %struct.btTypedObject, i32, i32, i8, %class.btRigidBody*, %class.btRigidBody*, float, float, %class.btVector3, %class.btVector3, %class.btVector3 }
%struct.btTypedObject = type { i32 }
%class.btHingeConstraint = type { %class.btTypedConstraint, [3 x %class.btJacobianEntry], [3 x %class.btJacobianEntry], %class.btTransform, %class.btTransform, float, float, float, float, float, float, float, float, float, float, float, float, float, i8, i8, i8, i8, i8, float }
%class.btJacobianEntry = type { %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, %class.btVector3, float }
%class.btConeTwistConstraint = type { %class.btTypedConstraint, [3 x %class.btJacobianEntry], %class.btTransform, %class.btTransform, float, float, float, float, float, float, float, float, %class.btVector3, %class.btVector3, float, float, float, float, float, float, float, float, i8, i8, i8, i8, float, float, %class.btVector3, i8, i8, %class.btQuaternion, float, %class.btVector3 }
%class.btQuaternion = type { %class.btQuadWord }
%class.btQuadWord = type { [4 x float] }

@_ZTV7RagDoll = external unnamed_addr constant [4 x i8*]

declare noalias i8* @_Znwm(i32)

declare i32 @__gxx_personality_sj0(...)

declare void @_ZdlPv(i8*) nounwind

declare %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3*, float*, float*, float*) unnamed_addr inlinehint ssp align 2

declare void @_ZSt9terminatev()

declare %class.btTransform* @_ZN11btTransformC1Ev(%class.btTransform*) unnamed_addr ssp align 2

declare void @_ZN11btTransform11setIdentityEv(%class.btTransform*) ssp align 2

declare void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform*, %class.btVector3*) nounwind inlinehint ssp align 2

declare i8* @_ZN13btConvexShapenwEm(i32) inlinehint ssp align 2

declare void @_ZN13btConvexShapedlEPv(i8*) inlinehint ssp align 2

declare %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape*, float, float)

declare %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform*) nounwind inlinehint ssp align 2

define %class.RagDoll* @_ZN7RagDollC2EP15btDynamicsWorldRK9btVector3f(%class.RagDoll* %this, %class.btDynamicsWorld* %ownerWorld, %class.btVector3* %positionOffset, float %scale) unnamed_addr ssp align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %retval = alloca %class.RagDoll*, align 4
  %this.addr = alloca %class.RagDoll*, align 4
  %ownerWorld.addr = alloca %class.btDynamicsWorld*, align 4
  %positionOffset.addr = alloca %class.btVector3*, align 4
  %scale.addr = alloca float, align 4
  %exn.slot = alloca i8*
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
  %hingeC = alloca %class.btHingeConstraint*, align 4
  %coneC = alloca %class.btConeTwistConstraint*, align 4
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
  store %class.RagDoll* %this, %class.RagDoll** %this.addr, align 4
  store %class.btDynamicsWorld* %ownerWorld, %class.btDynamicsWorld** %ownerWorld.addr, align 4
  store %class.btVector3* %positionOffset, %class.btVector3** %positionOffset.addr, align 4
  store float %scale, float* %scale.addr, align 4
  %this1 = load %class.RagDoll*, %class.RagDoll** %this.addr
  store %class.RagDoll* %this1, %class.RagDoll** %retval
  %0 = bitcast %class.RagDoll* %this1 to i8***
  store i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV7RagDoll, i64 0, i64 2), i8*** %0
  %m_ownerWorld = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %1 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %ownerWorld.addr, align 4
  store %class.btDynamicsWorld* %1, %class.btDynamicsWorld** %m_ownerWorld, align 4
  %call = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %2 = bitcast i8* %call to %class.btCapsuleShape*
  %3 = load float, float* %scale.addr, align 4
  %mul = fmul float 0x3FC3333340000000, %3
  %4 = load float, float* %scale.addr, align 4
  %mul2 = fmul float 0x3FC99999A0000000, %4
  %call3 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %2, float %mul, float %mul2)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %5 = bitcast %class.btCapsuleShape* %2 to %class.btCollisionShape*
  %m_shapes = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes, i32 0, i32 0
  store %class.btCollisionShape* %5, %class.btCollisionShape** %arrayidx, align 4
  %call5 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %6 = bitcast i8* %call5 to %class.btCapsuleShape*
  %7 = load float, float* %scale.addr, align 4
  %mul6 = fmul float 0x3FC3333340000000, %7
  %8 = load float, float* %scale.addr, align 4
  %mul7 = fmul float 0x3FD1EB8520000000, %8
  %call10 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %6, float %mul6, float %mul7)
          to label %invoke.cont9 unwind label %lpad8

invoke.cont9:                                     ; preds = %invoke.cont
  %9 = bitcast %class.btCapsuleShape* %6 to %class.btCollisionShape*
  %m_shapes12 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx13 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes12, i32 0, i32 1
  store %class.btCollisionShape* %9, %class.btCollisionShape** %arrayidx13, align 4
  %call14 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %10 = bitcast i8* %call14 to %class.btCapsuleShape*
  %11 = load float, float* %scale.addr, align 4
  %mul15 = fmul float 0x3FB99999A0000000, %11
  %12 = load float, float* %scale.addr, align 4
  %mul16 = fmul float 0x3FA99999A0000000, %12
  %call19 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %10, float %mul15, float %mul16)
          to label %invoke.cont18 unwind label %lpad17

invoke.cont18:                                    ; preds = %invoke.cont9
  %13 = bitcast %class.btCapsuleShape* %10 to %class.btCollisionShape*
  %m_shapes21 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx22 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes21, i32 0, i32 2
  store %class.btCollisionShape* %13, %class.btCollisionShape** %arrayidx22, align 4
  %call23 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %14 = bitcast i8* %call23 to %class.btCapsuleShape*
  %15 = load float, float* %scale.addr, align 4
  %mul24 = fmul float 0x3FB1EB8520000000, %15
  %16 = load float, float* %scale.addr, align 4
  %mul25 = fmul float 0x3FDCCCCCC0000000, %16
  %call28 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %14, float %mul24, float %mul25)
          to label %invoke.cont27 unwind label %lpad26

invoke.cont27:                                    ; preds = %invoke.cont18
  %17 = bitcast %class.btCapsuleShape* %14 to %class.btCollisionShape*
  %m_shapes30 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx31 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes30, i32 0, i32 3
  store %class.btCollisionShape* %17, %class.btCollisionShape** %arrayidx31, align 4
  %call32 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %18 = bitcast i8* %call32 to %class.btCapsuleShape*
  %19 = load float, float* %scale.addr, align 4
  %mul33 = fmul float 0x3FA99999A0000000, %19
  %20 = load float, float* %scale.addr, align 4
  %mul34 = fmul float 0x3FD7AE1480000000, %20
  %call37 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %18, float %mul33, float %mul34)
          to label %invoke.cont36 unwind label %lpad35

invoke.cont36:                                    ; preds = %invoke.cont27
  %21 = bitcast %class.btCapsuleShape* %18 to %class.btCollisionShape*
  %m_shapes39 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx40 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes39, i32 0, i32 4
  store %class.btCollisionShape* %21, %class.btCollisionShape** %arrayidx40, align 4
  %call41 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %22 = bitcast i8* %call41 to %class.btCapsuleShape*
  %23 = load float, float* %scale.addr, align 4
  %mul42 = fmul float 0x3FB1EB8520000000, %23
  %24 = load float, float* %scale.addr, align 4
  %mul43 = fmul float 0x3FDCCCCCC0000000, %24
  %call46 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %22, float %mul42, float %mul43)
          to label %invoke.cont45 unwind label %lpad44

invoke.cont45:                                    ; preds = %invoke.cont36
  %25 = bitcast %class.btCapsuleShape* %22 to %class.btCollisionShape*
  %m_shapes48 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx49 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes48, i32 0, i32 5
  store %class.btCollisionShape* %25, %class.btCollisionShape** %arrayidx49, align 4
  %call50 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %26 = bitcast i8* %call50 to %class.btCapsuleShape*
  %27 = load float, float* %scale.addr, align 4
  %mul51 = fmul float 0x3FA99999A0000000, %27
  %28 = load float, float* %scale.addr, align 4
  %mul52 = fmul float 0x3FD7AE1480000000, %28
  %call55 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %26, float %mul51, float %mul52)
          to label %invoke.cont54 unwind label %lpad53

invoke.cont54:                                    ; preds = %invoke.cont45
  %29 = bitcast %class.btCapsuleShape* %26 to %class.btCollisionShape*
  %m_shapes57 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx58 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes57, i32 0, i32 6
  store %class.btCollisionShape* %29, %class.btCollisionShape** %arrayidx58, align 4
  %call59 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %30 = bitcast i8* %call59 to %class.btCapsuleShape*
  %31 = load float, float* %scale.addr, align 4
  %mul60 = fmul float 0x3FA99999A0000000, %31
  %32 = load float, float* %scale.addr, align 4
  %mul61 = fmul float 0x3FD51EB860000000, %32
  %call64 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %30, float %mul60, float %mul61)
          to label %invoke.cont63 unwind label %lpad62

invoke.cont63:                                    ; preds = %invoke.cont54
  %33 = bitcast %class.btCapsuleShape* %30 to %class.btCollisionShape*
  %m_shapes66 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx67 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes66, i32 0, i32 7
  store %class.btCollisionShape* %33, %class.btCollisionShape** %arrayidx67, align 4
  %call68 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %34 = bitcast i8* %call68 to %class.btCapsuleShape*
  %35 = load float, float* %scale.addr, align 4
  %mul69 = fmul float 0x3FA47AE140000000, %35
  %36 = load float, float* %scale.addr, align 4
  %mul70 = fmul float 2.500000e-01, %36
  %call73 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %34, float %mul69, float %mul70)
          to label %invoke.cont72 unwind label %lpad71

invoke.cont72:                                    ; preds = %invoke.cont63
  %37 = bitcast %class.btCapsuleShape* %34 to %class.btCollisionShape*
  %m_shapes75 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx76 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes75, i32 0, i32 8
  store %class.btCollisionShape* %37, %class.btCollisionShape** %arrayidx76, align 4
  %call77 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %38 = bitcast i8* %call77 to %class.btCapsuleShape*
  %39 = load float, float* %scale.addr, align 4
  %mul78 = fmul float 0x3FA99999A0000000, %39
  %40 = load float, float* %scale.addr, align 4
  %mul79 = fmul float 0x3FD51EB860000000, %40
  %call82 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %38, float %mul78, float %mul79)
          to label %invoke.cont81 unwind label %lpad80

invoke.cont81:                                    ; preds = %invoke.cont72
  %41 = bitcast %class.btCapsuleShape* %38 to %class.btCollisionShape*
  %m_shapes84 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx85 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes84, i32 0, i32 9
  store %class.btCollisionShape* %41, %class.btCollisionShape** %arrayidx85, align 4
  %call86 = call i8* @_ZN13btConvexShapenwEm(i32 56)
  %42 = bitcast i8* %call86 to %class.btCapsuleShape*
  %43 = load float, float* %scale.addr, align 4
  %mul87 = fmul float 0x3FA47AE140000000, %43
  %44 = load float, float* %scale.addr, align 4
  %mul88 = fmul float 2.500000e-01, %44
  %call91 = invoke %class.btCapsuleShape* @_ZN14btCapsuleShapeC1Eff(%class.btCapsuleShape* %42, float %mul87, float %mul88)
          to label %invoke.cont90 unwind label %lpad89

invoke.cont90:                                    ; preds = %invoke.cont81
  %45 = bitcast %class.btCapsuleShape* %42 to %class.btCollisionShape*
  %m_shapes93 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx94 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes93, i32 0, i32 10
  store %class.btCollisionShape* %45, %class.btCollisionShape** %arrayidx94, align 4
  %call95 = call %class.btTransform* @_ZN11btTransformC1Ev(%class.btTransform* %offset)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %offset)
  %46 = load %class.btVector3*, %class.btVector3** %positionOffset.addr, align 4
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %offset, %class.btVector3* %46)
  %call96 = call %class.btTransform* @_ZN11btTransformC1Ev(%class.btTransform* %transform)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0.000000e+00, float* %ref.tmp98, align 4
  store float 1.000000e+00, float* %ref.tmp99, align 4
  store float 0.000000e+00, float* %ref.tmp100, align 4
  %call101 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp97, float* %ref.tmp98, float* %ref.tmp99, float* %ref.tmp100)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp, float* %scale.addr, %class.btVector3* %ref.tmp97)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp102, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes103 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx104 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes103, i32 0, i32 0
  %47 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx104, align 4
  %call105 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp102, %class.btCollisionShape* %47)
  %m_bodies = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx106 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies, i32 0, i32 0
  store %class.btRigidBody* %call105, %class.btRigidBody** %arrayidx106, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0.000000e+00, float* %ref.tmp109, align 4
  store float 0x3FF3333340000000, float* %ref.tmp110, align 4
  store float 0.000000e+00, float* %ref.tmp111, align 4
  %call112 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp108, float* %ref.tmp109, float* %ref.tmp110, float* %ref.tmp111)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp107, float* %scale.addr, %class.btVector3* %ref.tmp108)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp107)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp113, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes114 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx115 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes114, i32 0, i32 1
  %48 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx115, align 4
  %call116 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp113, %class.btCollisionShape* %48)
  %m_bodies117 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx118 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies117, i32 0, i32 1
  store %class.btRigidBody* %call116, %class.btRigidBody** %arrayidx118, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0.000000e+00, float* %ref.tmp121, align 4
  store float 0x3FF99999A0000000, float* %ref.tmp122, align 4
  store float 0.000000e+00, float* %ref.tmp123, align 4
  %call124 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp120, float* %ref.tmp121, float* %ref.tmp122, float* %ref.tmp123)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp119, float* %scale.addr, %class.btVector3* %ref.tmp120)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp119)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp125, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes126 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx127 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes126, i32 0, i32 2
  %49 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx127, align 4
  %call128 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp125, %class.btCollisionShape* %49)
  %m_bodies129 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx130 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies129, i32 0, i32 2
  store %class.btRigidBody* %call128, %class.btRigidBody** %arrayidx130, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0xBFC70A3D80000000, float* %ref.tmp133, align 4
  store float 0x3FE4CCCCC0000000, float* %ref.tmp134, align 4
  store float 0.000000e+00, float* %ref.tmp135, align 4
  %call136 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp132, float* %ref.tmp133, float* %ref.tmp134, float* %ref.tmp135)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp131, float* %scale.addr, %class.btVector3* %ref.tmp132)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp131)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp137, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes138 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx139 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes138, i32 0, i32 3
  %50 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx139, align 4
  %call140 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp137, %class.btCollisionShape* %50)
  %m_bodies141 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx142 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies141, i32 0, i32 3
  store %class.btRigidBody* %call140, %class.btRigidBody** %arrayidx142, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0xBFC70A3D80000000, float* %ref.tmp145, align 4
  store float 0x3FC99999A0000000, float* %ref.tmp146, align 4
  store float 0.000000e+00, float* %ref.tmp147, align 4
  %call148 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp144, float* %ref.tmp145, float* %ref.tmp146, float* %ref.tmp147)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp143, float* %scale.addr, %class.btVector3* %ref.tmp144)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp143)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp149, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes150 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx151 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes150, i32 0, i32 4
  %51 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx151, align 4
  %call152 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp149, %class.btCollisionShape* %51)
  %m_bodies153 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx154 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies153, i32 0, i32 4
  store %class.btRigidBody* %call152, %class.btRigidBody** %arrayidx154, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0x3FC70A3D80000000, float* %ref.tmp157, align 4
  store float 0x3FE4CCCCC0000000, float* %ref.tmp158, align 4
  store float 0.000000e+00, float* %ref.tmp159, align 4
  %call160 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp156, float* %ref.tmp157, float* %ref.tmp158, float* %ref.tmp159)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp155, float* %scale.addr, %class.btVector3* %ref.tmp156)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp155)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp161, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes162 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx163 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes162, i32 0, i32 5
  %52 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx163, align 4
  %call164 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp161, %class.btCollisionShape* %52)
  %m_bodies165 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx166 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies165, i32 0, i32 5
  store %class.btRigidBody* %call164, %class.btRigidBody** %arrayidx166, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0x3FC70A3D80000000, float* %ref.tmp169, align 4
  store float 0x3FC99999A0000000, float* %ref.tmp170, align 4
  store float 0.000000e+00, float* %ref.tmp171, align 4
  %call172 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp168, float* %ref.tmp169, float* %ref.tmp170, float* %ref.tmp171)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp167, float* %scale.addr, %class.btVector3* %ref.tmp168)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp167)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp173, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes174 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx175 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes174, i32 0, i32 6
  %53 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx175, align 4
  %call176 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp173, %class.btCollisionShape* %53)
  %m_bodies177 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx178 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies177, i32 0, i32 6
  store %class.btRigidBody* %call176, %class.btRigidBody** %arrayidx178, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0xBFD6666660000000, float* %ref.tmp181, align 4
  store float 0x3FF7333340000000, float* %ref.tmp182, align 4
  store float 0.000000e+00, float* %ref.tmp183, align 4
  %call184 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp180, float* %ref.tmp181, float* %ref.tmp182, float* %ref.tmp183)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp179, float* %scale.addr, %class.btVector3* %ref.tmp180)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp179)
  %call185 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call185, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp186, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes187 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx188 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes187, i32 0, i32 7
  %54 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx188, align 4
  %call189 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp186, %class.btCollisionShape* %54)
  %m_bodies190 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx191 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies190, i32 0, i32 7
  store %class.btRigidBody* %call189, %class.btRigidBody** %arrayidx191, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0xBFE6666660000000, float* %ref.tmp194, align 4
  store float 0x3FF7333340000000, float* %ref.tmp195, align 4
  store float 0.000000e+00, float* %ref.tmp196, align 4
  %call197 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp193, float* %ref.tmp194, float* %ref.tmp195, float* %ref.tmp196)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp192, float* %scale.addr, %class.btVector3* %ref.tmp193)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp192)
  %call198 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call198, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp199, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes200 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx201 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes200, i32 0, i32 8
  %55 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx201, align 4
  %call202 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp199, %class.btCollisionShape* %55)
  %m_bodies203 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx204 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies203, i32 0, i32 8
  store %class.btRigidBody* %call202, %class.btRigidBody** %arrayidx204, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0x3FD6666660000000, float* %ref.tmp207, align 4
  store float 0x3FF7333340000000, float* %ref.tmp208, align 4
  store float 0.000000e+00, float* %ref.tmp209, align 4
  %call210 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp206, float* %ref.tmp207, float* %ref.tmp208, float* %ref.tmp209)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp205, float* %scale.addr, %class.btVector3* %ref.tmp206)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp205)
  %call211 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call211, float 0.000000e+00, float 0.000000e+00, float 0xBFF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp212, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes213 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx214 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes213, i32 0, i32 9
  %56 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx214, align 4
  %call215 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp212, %class.btCollisionShape* %56)
  %m_bodies216 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx217 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies216, i32 0, i32 9
  store %class.btRigidBody* %call215, %class.btRigidBody** %arrayidx217, align 4
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %transform)
  store float 0x3FE6666660000000, float* %ref.tmp220, align 4
  store float 0x3FF7333340000000, float* %ref.tmp221, align 4
  store float 0.000000e+00, float* %ref.tmp222, align 4
  %call223 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp219, float* %ref.tmp220, float* %ref.tmp221, float* %ref.tmp222)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp218, float* %scale.addr, %class.btVector3* %ref.tmp219)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %transform, %class.btVector3* %ref.tmp218)
  %call224 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %transform)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call224, float 0.000000e+00, float 0.000000e+00, float 0xBFF921FB60000000)
  call void @_ZNK11btTransformmlERKS_(%class.btTransform* sret %ref.tmp225, %class.btTransform* %offset, %class.btTransform* %transform)
  %m_shapes226 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 2
  %arrayidx227 = getelementptr inbounds [11 x %class.btCollisionShape*], [11 x %class.btCollisionShape*]* %m_shapes226, i32 0, i32 10
  %57 = load %class.btCollisionShape*, %class.btCollisionShape** %arrayidx227, align 4
  %call228 = call %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll* %this1, float 1.000000e+00, %class.btTransform* %ref.tmp225, %class.btCollisionShape* %57)
  %m_bodies229 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx230 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies229, i32 0, i32 10
  store %class.btRigidBody* %call228, %class.btRigidBody** %arrayidx230, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %invoke.cont90
  %58 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %58, 11
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %59 = load i32, i32* %i, align 4
  %m_bodies231 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx232 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies231, i32 0, i32 %59
  %60 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx232, align 4
  call void @_ZN11btRigidBody10setDampingEff(%class.btRigidBody* %60, float 0x3FA99999A0000000, float 0x3FEB333340000000)
  %61 = load i32, i32* %i, align 4
  %m_bodies233 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx234 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies233, i32 0, i32 %61
  %62 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx234, align 4
  %63 = bitcast %class.btRigidBody* %62 to %class.btCollisionObject*
  call void @_ZN17btCollisionObject19setDeactivationTimeEf(%class.btCollisionObject* %63, float 0x3FE99999A0000000)
  %64 = load i32, i32* %i, align 4
  %m_bodies235 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx236 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies235, i32 0, i32 %64
  %65 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx236, align 4
  call void @_ZN11btRigidBody21setSleepingThresholdsEff(%class.btRigidBody* %65, float 0x3FF99999A0000000, float 2.500000e+00)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %66 = load i32, i32* %i, align 4
  %inc = add nsw i32 %66, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

lpad:                                             ; preds = %entry
  %67 = landingpad { i8*, i32 }
          cleanup
  %68 = extractvalue { i8*, i32 } %67, 0
  store i8* %68, i8** %exn.slot
  %69 = extractvalue { i8*, i32 } %67, 1
  store i32 %69, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call)
          to label %invoke.cont4 unwind label %terminate.lpad

invoke.cont4:                                     ; preds = %lpad
  br label %eh.resume

lpad8:                                            ; preds = %invoke.cont
  %70 = landingpad { i8*, i32 }
          cleanup
  %71 = extractvalue { i8*, i32 } %70, 0
  store i8* %71, i8** %exn.slot
  %72 = extractvalue { i8*, i32 } %70, 1
  store i32 %72, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call5)
          to label %invoke.cont11 unwind label %terminate.lpad

invoke.cont11:                                    ; preds = %lpad8
  br label %eh.resume

lpad17:                                           ; preds = %invoke.cont9
  %73 = landingpad { i8*, i32 }
          cleanup
  %74 = extractvalue { i8*, i32 } %73, 0
  store i8* %74, i8** %exn.slot
  %75 = extractvalue { i8*, i32 } %73, 1
  store i32 %75, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call14)
          to label %invoke.cont20 unwind label %terminate.lpad

invoke.cont20:                                    ; preds = %lpad17
  br label %eh.resume

lpad26:                                           ; preds = %invoke.cont18
  %76 = landingpad { i8*, i32 }
          cleanup
  %77 = extractvalue { i8*, i32 } %76, 0
  store i8* %77, i8** %exn.slot
  %78 = extractvalue { i8*, i32 } %76, 1
  store i32 %78, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call23)
          to label %invoke.cont29 unwind label %terminate.lpad

invoke.cont29:                                    ; preds = %lpad26
  br label %eh.resume

lpad35:                                           ; preds = %invoke.cont27
  %79 = landingpad { i8*, i32 }
          cleanup
  %80 = extractvalue { i8*, i32 } %79, 0
  store i8* %80, i8** %exn.slot
  %81 = extractvalue { i8*, i32 } %79, 1
  store i32 %81, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call32)
          to label %invoke.cont38 unwind label %terminate.lpad

invoke.cont38:                                    ; preds = %lpad35
  br label %eh.resume

lpad44:                                           ; preds = %invoke.cont36
  %82 = landingpad { i8*, i32 }
          cleanup
  %83 = extractvalue { i8*, i32 } %82, 0
  store i8* %83, i8** %exn.slot
  %84 = extractvalue { i8*, i32 } %82, 1
  store i32 %84, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call41)
          to label %invoke.cont47 unwind label %terminate.lpad

invoke.cont47:                                    ; preds = %lpad44
  br label %eh.resume

lpad53:                                           ; preds = %invoke.cont45
  %85 = landingpad { i8*, i32 }
          cleanup
  %86 = extractvalue { i8*, i32 } %85, 0
  store i8* %86, i8** %exn.slot
  %87 = extractvalue { i8*, i32 } %85, 1
  store i32 %87, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call50)
          to label %invoke.cont56 unwind label %terminate.lpad

invoke.cont56:                                    ; preds = %lpad53
  br label %eh.resume

lpad62:                                           ; preds = %invoke.cont54
  %88 = landingpad { i8*, i32 }
          cleanup
  %89 = extractvalue { i8*, i32 } %88, 0
  store i8* %89, i8** %exn.slot
  %90 = extractvalue { i8*, i32 } %88, 1
  store i32 %90, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call59)
          to label %invoke.cont65 unwind label %terminate.lpad

invoke.cont65:                                    ; preds = %lpad62
  br label %eh.resume

lpad71:                                           ; preds = %invoke.cont63
  %91 = landingpad { i8*, i32 }
          cleanup
  %92 = extractvalue { i8*, i32 } %91, 0
  store i8* %92, i8** %exn.slot
  %93 = extractvalue { i8*, i32 } %91, 1
  store i32 %93, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call68)
          to label %invoke.cont74 unwind label %terminate.lpad

invoke.cont74:                                    ; preds = %lpad71
  br label %eh.resume

lpad80:                                           ; preds = %invoke.cont72
  %94 = landingpad { i8*, i32 }
          cleanup
  %95 = extractvalue { i8*, i32 } %94, 0
  store i8* %95, i8** %exn.slot
  %96 = extractvalue { i8*, i32 } %94, 1
  store i32 %96, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call77)
          to label %invoke.cont83 unwind label %terminate.lpad

invoke.cont83:                                    ; preds = %lpad80
  br label %eh.resume

lpad89:                                           ; preds = %invoke.cont81
  %97 = landingpad { i8*, i32 }
          cleanup
  %98 = extractvalue { i8*, i32 } %97, 0
  store i8* %98, i8** %exn.slot
  %99 = extractvalue { i8*, i32 } %97, 1
  store i32 %99, i32* %ehselector.slot
  invoke void @_ZN13btConvexShapedlEPv(i8* %call86)
          to label %invoke.cont92 unwind label %terminate.lpad

invoke.cont92:                                    ; preds = %lpad89
  br label %eh.resume

for.end:                                          ; preds = %for.cond
  %call237 = call %class.btTransform* @_ZN11btTransformC1Ev(%class.btTransform* %localA)
  %call238 = call %class.btTransform* @_ZN11btTransformC1Ev(%class.btTransform* %localB)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call239 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call239, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp242, align 4
  store float 0x3FC3333340000000, float* %ref.tmp243, align 4
  store float 0.000000e+00, float* %ref.tmp244, align 4
  %call245 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp241, float* %ref.tmp242, float* %ref.tmp243, float* %ref.tmp244)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp240, float* %scale.addr, %class.btVector3* %ref.tmp241)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp240)
  %call246 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call246, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp249, align 4
  store float 0xBFC3333340000000, float* %ref.tmp250, align 4
  store float 0.000000e+00, float* %ref.tmp251, align 4
  %call252 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp248, float* %ref.tmp249, float* %ref.tmp250, float* %ref.tmp251)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp247, float* %scale.addr, %class.btVector3* %ref.tmp248)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp247)
  %call253 = call noalias i8* @_Znwm(i32 780)
  %100 = bitcast i8* %call253 to %class.btHingeConstraint*
  %m_bodies254 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx255 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies254, i32 0, i32 0
  %101 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx255, align 4
  %m_bodies256 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx257 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies256, i32 0, i32 1
  %102 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx257, align 4
  %call260 = invoke %class.btHingeConstraint* @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(%class.btHingeConstraint* %100, %class.btRigidBody* %101, %class.btRigidBody* %102, %class.btTransform* %localA, %class.btTransform* %localB, i1 zeroext false)
          to label %invoke.cont259 unwind label %lpad258

invoke.cont259:                                   ; preds = %for.end
  store %class.btHingeConstraint* %100, %class.btHingeConstraint** %hingeC, align 4
  %103 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(%class.btHingeConstraint* %103, float 0xBFE921FB60000000, float 0x3FF921FB60000000, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %104 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  %105 = bitcast %class.btHingeConstraint* %104 to %class.btTypedConstraint*
  %m_joints = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx261 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints, i32 0, i32 0
  store %class.btTypedConstraint* %105, %class.btTypedConstraint** %arrayidx261, align 4
  %m_ownerWorld262 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %106 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld262, align 4
  %107 = bitcast %class.btDynamicsWorld* %106 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %107
  %vfn = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable, i64 10
  %108 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn
  %m_joints263 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx264 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints263, i32 0, i32 0
  %109 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx264, align 4
  call void %108(%class.btDynamicsWorld* %106, %class.btTypedConstraint* %109, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call265 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call265, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, float* %ref.tmp268, align 4
  store float 0x3FD3333340000000, float* %ref.tmp269, align 4
  store float 0.000000e+00, float* %ref.tmp270, align 4
  %call271 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp267, float* %ref.tmp268, float* %ref.tmp269, float* %ref.tmp270)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp266, float* %scale.addr, %class.btVector3* %ref.tmp267)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp266)
  %call272 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call272, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, float* %ref.tmp275, align 4
  store float 0xBFC1EB8520000000, float* %ref.tmp276, align 4
  store float 0.000000e+00, float* %ref.tmp277, align 4
  %call278 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp274, float* %ref.tmp275, float* %ref.tmp276, float* %ref.tmp277)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp273, float* %scale.addr, %class.btVector3* %ref.tmp274)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp273)
  %call279 = call noalias i8* @_Znwm(i32 628)
  %110 = bitcast i8* %call279 to %class.btConeTwistConstraint*
  %m_bodies280 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx281 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies280, i32 0, i32 1
  %111 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx281, align 4
  %m_bodies282 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx283 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies282, i32 0, i32 2
  %112 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx283, align 4
  %call286 = invoke %class.btConeTwistConstraint* @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(%class.btConeTwistConstraint* %110, %class.btRigidBody* %111, %class.btRigidBody* %112, %class.btTransform* %localA, %class.btTransform* %localB)
          to label %invoke.cont285 unwind label %lpad284

invoke.cont285:                                   ; preds = %invoke.cont259
  store %class.btConeTwistConstraint* %110, %class.btConeTwistConstraint** %coneC, align 4
  %113 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(%class.btConeTwistConstraint* %113, float 0x3FE921FB60000000, float 0x3FE921FB60000000, float 0x3FF921FB60000000, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %114 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  %115 = bitcast %class.btConeTwistConstraint* %114 to %class.btTypedConstraint*
  %m_joints287 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx288 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints287, i32 0, i32 1
  store %class.btTypedConstraint* %115, %class.btTypedConstraint** %arrayidx288, align 4
  %m_ownerWorld289 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %116 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld289, align 4
  %117 = bitcast %class.btDynamicsWorld* %116 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable290 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %117
  %vfn291 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable290, i64 10
  %118 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn291
  %m_joints292 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx293 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints292, i32 0, i32 1
  %119 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx293, align 4
  call void %118(%class.btDynamicsWorld* %116, %class.btTypedConstraint* %119, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call294 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call294, float 0.000000e+00, float 0.000000e+00, float 0xC00F6A7A20000000)
  store float 0xBFC70A3D80000000, float* %ref.tmp297, align 4
  store float 0xBFB99999A0000000, float* %ref.tmp298, align 4
  store float 0.000000e+00, float* %ref.tmp299, align 4
  %call300 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp296, float* %ref.tmp297, float* %ref.tmp298, float* %ref.tmp299)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp295, float* %scale.addr, %class.btVector3* %ref.tmp296)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp295)
  %call301 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call301, float 0.000000e+00, float 0.000000e+00, float 0xC00F6A7A20000000)
  store float 0.000000e+00, float* %ref.tmp304, align 4
  store float 0x3FCCCCCCC0000000, float* %ref.tmp305, align 4
  store float 0.000000e+00, float* %ref.tmp306, align 4
  %call307 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp303, float* %ref.tmp304, float* %ref.tmp305, float* %ref.tmp306)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp302, float* %scale.addr, %class.btVector3* %ref.tmp303)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp302)
  %call308 = call noalias i8* @_Znwm(i32 628)
  %120 = bitcast i8* %call308 to %class.btConeTwistConstraint*
  %m_bodies309 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx310 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies309, i32 0, i32 0
  %121 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx310, align 4
  %m_bodies311 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx312 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies311, i32 0, i32 3
  %122 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx312, align 4
  %call315 = invoke %class.btConeTwistConstraint* @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(%class.btConeTwistConstraint* %120, %class.btRigidBody* %121, %class.btRigidBody* %122, %class.btTransform* %localA, %class.btTransform* %localB)
          to label %invoke.cont314 unwind label %lpad313

invoke.cont314:                                   ; preds = %invoke.cont285
  store %class.btConeTwistConstraint* %120, %class.btConeTwistConstraint** %coneC, align 4
  %123 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(%class.btConeTwistConstraint* %123, float 0x3FE921FB60000000, float 0x3FE921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %124 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  %125 = bitcast %class.btConeTwistConstraint* %124 to %class.btTypedConstraint*
  %m_joints316 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx317 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints316, i32 0, i32 2
  store %class.btTypedConstraint* %125, %class.btTypedConstraint** %arrayidx317, align 4
  %m_ownerWorld318 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %126 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld318, align 4
  %127 = bitcast %class.btDynamicsWorld* %126 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable319 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %127
  %vfn320 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable319, i64 10
  %128 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn320
  %m_joints321 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx322 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints321, i32 0, i32 2
  %129 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx322, align 4
  call void %128(%class.btDynamicsWorld* %126, %class.btTypedConstraint* %129, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call323 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call323, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp326, align 4
  store float 0xBFCCCCCCC0000000, float* %ref.tmp327, align 4
  store float 0.000000e+00, float* %ref.tmp328, align 4
  %call329 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp325, float* %ref.tmp326, float* %ref.tmp327, float* %ref.tmp328)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp324, float* %scale.addr, %class.btVector3* %ref.tmp325)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp324)
  %call330 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call330, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp333, align 4
  store float 0x3FC7AE1480000000, float* %ref.tmp334, align 4
  store float 0.000000e+00, float* %ref.tmp335, align 4
  %call336 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp332, float* %ref.tmp333, float* %ref.tmp334, float* %ref.tmp335)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp331, float* %scale.addr, %class.btVector3* %ref.tmp332)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp331)
  %call337 = call noalias i8* @_Znwm(i32 780)
  %130 = bitcast i8* %call337 to %class.btHingeConstraint*
  %m_bodies338 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx339 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies338, i32 0, i32 3
  %131 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx339, align 4
  %m_bodies340 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx341 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies340, i32 0, i32 4
  %132 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx341, align 4
  %call344 = invoke %class.btHingeConstraint* @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(%class.btHingeConstraint* %130, %class.btRigidBody* %131, %class.btRigidBody* %132, %class.btTransform* %localA, %class.btTransform* %localB, i1 zeroext false)
          to label %invoke.cont343 unwind label %lpad342

invoke.cont343:                                   ; preds = %invoke.cont314
  store %class.btHingeConstraint* %130, %class.btHingeConstraint** %hingeC, align 4
  %133 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(%class.btHingeConstraint* %133, float 0.000000e+00, float 0x3FF921FB60000000, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %134 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  %135 = bitcast %class.btHingeConstraint* %134 to %class.btTypedConstraint*
  %m_joints345 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx346 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints345, i32 0, i32 3
  store %class.btTypedConstraint* %135, %class.btTypedConstraint** %arrayidx346, align 4
  %m_ownerWorld347 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %136 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld347, align 4
  %137 = bitcast %class.btDynamicsWorld* %136 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable348 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %137
  %vfn349 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable348, i64 10
  %138 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn349
  %m_joints350 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx351 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints350, i32 0, i32 3
  %139 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx351, align 4
  call void %138(%class.btDynamicsWorld* %136, %class.btTypedConstraint* %139, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call352 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call352, float 0.000000e+00, float 0.000000e+00, float 0x3FE921FB60000000)
  store float 0x3FC70A3D80000000, float* %ref.tmp355, align 4
  store float 0xBFB99999A0000000, float* %ref.tmp356, align 4
  store float 0.000000e+00, float* %ref.tmp357, align 4
  %call358 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp354, float* %ref.tmp355, float* %ref.tmp356, float* %ref.tmp357)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp353, float* %scale.addr, %class.btVector3* %ref.tmp354)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp353)
  %call359 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call359, float 0.000000e+00, float 0.000000e+00, float 0x3FE921FB60000000)
  store float 0.000000e+00, float* %ref.tmp362, align 4
  store float 0x3FCCCCCCC0000000, float* %ref.tmp363, align 4
  store float 0.000000e+00, float* %ref.tmp364, align 4
  %call365 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp361, float* %ref.tmp362, float* %ref.tmp363, float* %ref.tmp364)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp360, float* %scale.addr, %class.btVector3* %ref.tmp361)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp360)
  %call366 = call noalias i8* @_Znwm(i32 628)
  %140 = bitcast i8* %call366 to %class.btConeTwistConstraint*
  %m_bodies367 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx368 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies367, i32 0, i32 0
  %141 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx368, align 4
  %m_bodies369 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx370 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies369, i32 0, i32 5
  %142 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx370, align 4
  %call373 = invoke %class.btConeTwistConstraint* @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(%class.btConeTwistConstraint* %140, %class.btRigidBody* %141, %class.btRigidBody* %142, %class.btTransform* %localA, %class.btTransform* %localB)
          to label %invoke.cont372 unwind label %lpad371

invoke.cont372:                                   ; preds = %invoke.cont343
  store %class.btConeTwistConstraint* %140, %class.btConeTwistConstraint** %coneC, align 4
  %143 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(%class.btConeTwistConstraint* %143, float 0x3FE921FB60000000, float 0x3FE921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %144 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  %145 = bitcast %class.btConeTwistConstraint* %144 to %class.btTypedConstraint*
  %m_joints374 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx375 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints374, i32 0, i32 4
  store %class.btTypedConstraint* %145, %class.btTypedConstraint** %arrayidx375, align 4
  %m_ownerWorld376 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %146 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld376, align 4
  %147 = bitcast %class.btDynamicsWorld* %146 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable377 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %147
  %vfn378 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable377, i64 10
  %148 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn378
  %m_joints379 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx380 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints379, i32 0, i32 4
  %149 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx380, align 4
  call void %148(%class.btDynamicsWorld* %146, %class.btTypedConstraint* %149, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call381 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call381, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp384, align 4
  store float 0xBFCCCCCCC0000000, float* %ref.tmp385, align 4
  store float 0.000000e+00, float* %ref.tmp386, align 4
  %call387 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp383, float* %ref.tmp384, float* %ref.tmp385, float* %ref.tmp386)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp382, float* %scale.addr, %class.btVector3* %ref.tmp383)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp382)
  %call388 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call388, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp391, align 4
  store float 0x3FC7AE1480000000, float* %ref.tmp392, align 4
  store float 0.000000e+00, float* %ref.tmp393, align 4
  %call394 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp390, float* %ref.tmp391, float* %ref.tmp392, float* %ref.tmp393)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp389, float* %scale.addr, %class.btVector3* %ref.tmp390)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp389)
  %call395 = call noalias i8* @_Znwm(i32 780)
  %150 = bitcast i8* %call395 to %class.btHingeConstraint*
  %m_bodies396 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx397 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies396, i32 0, i32 5
  %151 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx397, align 4
  %m_bodies398 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx399 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies398, i32 0, i32 6
  %152 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx399, align 4
  %call402 = invoke %class.btHingeConstraint* @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(%class.btHingeConstraint* %150, %class.btRigidBody* %151, %class.btRigidBody* %152, %class.btTransform* %localA, %class.btTransform* %localB, i1 zeroext false)
          to label %invoke.cont401 unwind label %lpad400

invoke.cont401:                                   ; preds = %invoke.cont372
  store %class.btHingeConstraint* %150, %class.btHingeConstraint** %hingeC, align 4
  %153 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(%class.btHingeConstraint* %153, float 0.000000e+00, float 0x3FF921FB60000000, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %154 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  %155 = bitcast %class.btHingeConstraint* %154 to %class.btTypedConstraint*
  %m_joints403 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx404 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints403, i32 0, i32 5
  store %class.btTypedConstraint* %155, %class.btTypedConstraint** %arrayidx404, align 4
  %m_ownerWorld405 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %156 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld405, align 4
  %157 = bitcast %class.btDynamicsWorld* %156 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable406 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %157
  %vfn407 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable406, i64 10
  %158 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn407
  %m_joints408 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx409 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints408, i32 0, i32 5
  %159 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx409, align 4
  call void %158(%class.btDynamicsWorld* %156, %class.btTypedConstraint* %159, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call410 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call410, float 0.000000e+00, float 0.000000e+00, float 0x400921FB60000000)
  store float 0xBFC99999A0000000, float* %ref.tmp413, align 4
  store float 0x3FC3333340000000, float* %ref.tmp414, align 4
  store float 0.000000e+00, float* %ref.tmp415, align 4
  %call416 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp412, float* %ref.tmp413, float* %ref.tmp414, float* %ref.tmp415)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp411, float* %scale.addr, %class.btVector3* %ref.tmp412)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp411)
  %call417 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call417, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, float* %ref.tmp420, align 4
  store float 0xBFC70A3D80000000, float* %ref.tmp421, align 4
  store float 0.000000e+00, float* %ref.tmp422, align 4
  %call423 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp419, float* %ref.tmp420, float* %ref.tmp421, float* %ref.tmp422)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp418, float* %scale.addr, %class.btVector3* %ref.tmp419)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp418)
  %call424 = call noalias i8* @_Znwm(i32 628)
  %160 = bitcast i8* %call424 to %class.btConeTwistConstraint*
  %m_bodies425 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx426 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies425, i32 0, i32 1
  %161 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx426, align 4
  %m_bodies427 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx428 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies427, i32 0, i32 7
  %162 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx428, align 4
  %call431 = invoke %class.btConeTwistConstraint* @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(%class.btConeTwistConstraint* %160, %class.btRigidBody* %161, %class.btRigidBody* %162, %class.btTransform* %localA, %class.btTransform* %localB)
          to label %invoke.cont430 unwind label %lpad429

invoke.cont430:                                   ; preds = %invoke.cont401
  store %class.btConeTwistConstraint* %160, %class.btConeTwistConstraint** %coneC, align 4
  %163 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(%class.btConeTwistConstraint* %163, float 0x3FF921FB60000000, float 0x3FF921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %164 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  %165 = bitcast %class.btConeTwistConstraint* %164 to %class.btTypedConstraint*
  %m_joints432 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx433 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints432, i32 0, i32 6
  store %class.btTypedConstraint* %165, %class.btTypedConstraint** %arrayidx433, align 4
  %m_ownerWorld434 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %166 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld434, align 4
  %167 = bitcast %class.btDynamicsWorld* %166 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable435 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %167
  %vfn436 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable435, i64 10
  %168 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn436
  %m_joints437 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx438 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints437, i32 0, i32 6
  %169 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx438, align 4
  call void %168(%class.btDynamicsWorld* %166, %class.btTypedConstraint* %169, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call439 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call439, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp442, align 4
  store float 0x3FC70A3D80000000, float* %ref.tmp443, align 4
  store float 0.000000e+00, float* %ref.tmp444, align 4
  %call445 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp441, float* %ref.tmp442, float* %ref.tmp443, float* %ref.tmp444)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp440, float* %scale.addr, %class.btVector3* %ref.tmp441)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp440)
  %call446 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call446, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp449, align 4
  store float 0xBFC1EB8520000000, float* %ref.tmp450, align 4
  store float 0.000000e+00, float* %ref.tmp451, align 4
  %call452 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp448, float* %ref.tmp449, float* %ref.tmp450, float* %ref.tmp451)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp447, float* %scale.addr, %class.btVector3* %ref.tmp448)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp447)
  %call453 = call noalias i8* @_Znwm(i32 780)
  %170 = bitcast i8* %call453 to %class.btHingeConstraint*
  %m_bodies454 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx455 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies454, i32 0, i32 7
  %171 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx455, align 4
  %m_bodies456 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx457 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies456, i32 0, i32 8
  %172 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx457, align 4
  %call460 = invoke %class.btHingeConstraint* @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(%class.btHingeConstraint* %170, %class.btRigidBody* %171, %class.btRigidBody* %172, %class.btTransform* %localA, %class.btTransform* %localB, i1 zeroext false)
          to label %invoke.cont459 unwind label %lpad458

invoke.cont459:                                   ; preds = %invoke.cont430
  store %class.btHingeConstraint* %170, %class.btHingeConstraint** %hingeC, align 4
  %173 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(%class.btHingeConstraint* %173, float 0xBFF921FB60000000, float 0.000000e+00, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %174 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  %175 = bitcast %class.btHingeConstraint* %174 to %class.btTypedConstraint*
  %m_joints461 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx462 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints461, i32 0, i32 7
  store %class.btTypedConstraint* %175, %class.btTypedConstraint** %arrayidx462, align 4
  %m_ownerWorld463 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %176 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld463, align 4
  %177 = bitcast %class.btDynamicsWorld* %176 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable464 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %177
  %vfn465 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable464, i64 10
  %178 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn465
  %m_joints466 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx467 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints466, i32 0, i32 7
  %179 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx467, align 4
  call void %178(%class.btDynamicsWorld* %176, %class.btTypedConstraint* %179, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call468 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call468, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
  store float 0x3FC99999A0000000, float* %ref.tmp471, align 4
  store float 0x3FC3333340000000, float* %ref.tmp472, align 4
  store float 0.000000e+00, float* %ref.tmp473, align 4
  %call474 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp470, float* %ref.tmp471, float* %ref.tmp472, float* %ref.tmp473)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp469, float* %scale.addr, %class.btVector3* %ref.tmp470)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp469)
  %call475 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call475, float 0.000000e+00, float 0.000000e+00, float 0x3FF921FB60000000)
  store float 0.000000e+00, float* %ref.tmp478, align 4
  store float 0xBFC70A3D80000000, float* %ref.tmp479, align 4
  store float 0.000000e+00, float* %ref.tmp480, align 4
  %call481 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp477, float* %ref.tmp478, float* %ref.tmp479, float* %ref.tmp480)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp476, float* %scale.addr, %class.btVector3* %ref.tmp477)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp476)
  %call482 = call noalias i8* @_Znwm(i32 628)
  %180 = bitcast i8* %call482 to %class.btConeTwistConstraint*
  %m_bodies483 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx484 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies483, i32 0, i32 1
  %181 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx484, align 4
  %m_bodies485 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx486 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies485, i32 0, i32 9
  %182 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx486, align 4
  %call489 = invoke %class.btConeTwistConstraint* @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(%class.btConeTwistConstraint* %180, %class.btRigidBody* %181, %class.btRigidBody* %182, %class.btTransform* %localA, %class.btTransform* %localB)
          to label %invoke.cont488 unwind label %lpad487

invoke.cont488:                                   ; preds = %invoke.cont459
  store %class.btConeTwistConstraint* %180, %class.btConeTwistConstraint** %coneC, align 4
  %183 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  call void @_ZN21btConeTwistConstraint8setLimitEffffff(%class.btConeTwistConstraint* %183, float 0x3FF921FB60000000, float 0x3FF921FB60000000, float 0.000000e+00, float 1.000000e+00, float 0x3FD3333340000000, float 1.000000e+00)
  %184 = load %class.btConeTwistConstraint*, %class.btConeTwistConstraint** %coneC, align 4
  %185 = bitcast %class.btConeTwistConstraint* %184 to %class.btTypedConstraint*
  %m_joints490 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx491 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints490, i32 0, i32 8
  store %class.btTypedConstraint* %185, %class.btTypedConstraint** %arrayidx491, align 4
  %m_ownerWorld492 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %186 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld492, align 4
  %187 = bitcast %class.btDynamicsWorld* %186 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable493 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %187
  %vfn494 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable493, i64 10
  %188 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn494
  %m_joints495 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx496 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints495, i32 0, i32 8
  %189 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx496, align 4
  call void %188(%class.btDynamicsWorld* %186, %class.btTypedConstraint* %189, i1 zeroext true)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localA)
  call void @_ZN11btTransform11setIdentityEv(%class.btTransform* %localB)
  %call497 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localA)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call497, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp500, align 4
  store float 0x3FC70A3D80000000, float* %ref.tmp501, align 4
  store float 0.000000e+00, float* %ref.tmp502, align 4
  %call503 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp499, float* %ref.tmp500, float* %ref.tmp501, float* %ref.tmp502)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp498, float* %scale.addr, %class.btVector3* %ref.tmp499)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localA, %class.btVector3* %ref.tmp498)
  %call504 = call %class.btMatrix3x3* @_ZN11btTransform8getBasisEv(%class.btTransform* %localB)
  call void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3* %call504, float 0.000000e+00, float 0x3FF921FB60000000, float 0.000000e+00)
  store float 0.000000e+00, float* %ref.tmp507, align 4
  store float 0xBFC1EB8520000000, float* %ref.tmp508, align 4
  store float 0.000000e+00, float* %ref.tmp509, align 4
  %call510 = call %class.btVector3* @_ZN9btVector3C1ERKfS1_S1_(%class.btVector3* %ref.tmp506, float* %ref.tmp507, float* %ref.tmp508, float* %ref.tmp509)
  call void @_ZmlRKfRK9btVector3(%class.btVector3* sret %ref.tmp505, float* %scale.addr, %class.btVector3* %ref.tmp506)
  call void @_ZN11btTransform9setOriginERK9btVector3(%class.btTransform* %localB, %class.btVector3* %ref.tmp505)
  %call511 = call noalias i8* @_Znwm(i32 780)
  %190 = bitcast i8* %call511 to %class.btHingeConstraint*
  %m_bodies512 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx513 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies512, i32 0, i32 9
  %191 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx513, align 4
  %m_bodies514 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 3
  %arrayidx515 = getelementptr inbounds [11 x %class.btRigidBody*], [11 x %class.btRigidBody*]* %m_bodies514, i32 0, i32 10
  %192 = load %class.btRigidBody*, %class.btRigidBody** %arrayidx515, align 4
  %call518 = invoke %class.btHingeConstraint* @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(%class.btHingeConstraint* %190, %class.btRigidBody* %191, %class.btRigidBody* %192, %class.btTransform* %localA, %class.btTransform* %localB, i1 zeroext false)
          to label %invoke.cont517 unwind label %lpad516

invoke.cont517:                                   ; preds = %invoke.cont488
  store %class.btHingeConstraint* %190, %class.btHingeConstraint** %hingeC, align 4
  %193 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  call void @_ZN17btHingeConstraint8setLimitEfffff(%class.btHingeConstraint* %193, float 0xBFF921FB60000000, float 0.000000e+00, float 0x3FECCCCCC0000000, float 0x3FD3333340000000, float 1.000000e+00)
  %194 = load %class.btHingeConstraint*, %class.btHingeConstraint** %hingeC, align 4
  %195 = bitcast %class.btHingeConstraint* %194 to %class.btTypedConstraint*
  %m_joints519 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx520 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints519, i32 0, i32 9
  store %class.btTypedConstraint* %195, %class.btTypedConstraint** %arrayidx520, align 4
  %m_ownerWorld521 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 1
  %196 = load %class.btDynamicsWorld*, %class.btDynamicsWorld** %m_ownerWorld521, align 4
  %197 = bitcast %class.btDynamicsWorld* %196 to void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)***
  %vtable522 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)**, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*** %197
  %vfn523 = getelementptr inbounds void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vtable522, i64 10
  %198 = load void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)*, void (%class.btDynamicsWorld*, %class.btTypedConstraint*, i1)** %vfn523
  %m_joints524 = getelementptr inbounds %class.RagDoll, %class.RagDoll* %this1, i32 0, i32 4
  %arrayidx525 = getelementptr inbounds [10 x %class.btTypedConstraint*], [10 x %class.btTypedConstraint*]* %m_joints524, i32 0, i32 9
  %199 = load %class.btTypedConstraint*, %class.btTypedConstraint** %arrayidx525, align 4
  call void %198(%class.btDynamicsWorld* %196, %class.btTypedConstraint* %199, i1 zeroext true)
  %200 = load %class.RagDoll*, %class.RagDoll** %retval
  ret %class.RagDoll* %200

lpad258:                                          ; preds = %for.end
  %201 = landingpad { i8*, i32 }
          cleanup
  %202 = extractvalue { i8*, i32 } %201, 0
  store i8* %202, i8** %exn.slot
  %203 = extractvalue { i8*, i32 } %201, 1
  store i32 %203, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call253) nounwind
  br label %eh.resume

lpad284:                                          ; preds = %invoke.cont259
  %204 = landingpad { i8*, i32 }
          cleanup
  %205 = extractvalue { i8*, i32 } %204, 0
  store i8* %205, i8** %exn.slot
  %206 = extractvalue { i8*, i32 } %204, 1
  store i32 %206, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call279) nounwind
  br label %eh.resume

lpad313:                                          ; preds = %invoke.cont285
  %207 = landingpad { i8*, i32 }
          cleanup
  %208 = extractvalue { i8*, i32 } %207, 0
  store i8* %208, i8** %exn.slot
  %209 = extractvalue { i8*, i32 } %207, 1
  store i32 %209, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call308) nounwind
  br label %eh.resume

lpad342:                                          ; preds = %invoke.cont314
  %210 = landingpad { i8*, i32 }
          cleanup
  %211 = extractvalue { i8*, i32 } %210, 0
  store i8* %211, i8** %exn.slot
  %212 = extractvalue { i8*, i32 } %210, 1
  store i32 %212, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call337) nounwind
  br label %eh.resume

lpad371:                                          ; preds = %invoke.cont343
  %213 = landingpad { i8*, i32 }
          cleanup
  %214 = extractvalue { i8*, i32 } %213, 0
  store i8* %214, i8** %exn.slot
  %215 = extractvalue { i8*, i32 } %213, 1
  store i32 %215, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call366) nounwind
  br label %eh.resume

lpad400:                                          ; preds = %invoke.cont372
  %216 = landingpad { i8*, i32 }
          cleanup
  %217 = extractvalue { i8*, i32 } %216, 0
  store i8* %217, i8** %exn.slot
  %218 = extractvalue { i8*, i32 } %216, 1
  store i32 %218, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call395) nounwind
  br label %eh.resume

lpad429:                                          ; preds = %invoke.cont401
  %219 = landingpad { i8*, i32 }
          cleanup
  %220 = extractvalue { i8*, i32 } %219, 0
  store i8* %220, i8** %exn.slot
  %221 = extractvalue { i8*, i32 } %219, 1
  store i32 %221, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call424) nounwind
  br label %eh.resume

lpad458:                                          ; preds = %invoke.cont430
  %222 = landingpad { i8*, i32 }
          cleanup
  %223 = extractvalue { i8*, i32 } %222, 0
  store i8* %223, i8** %exn.slot
  %224 = extractvalue { i8*, i32 } %222, 1
  store i32 %224, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call453) nounwind
  br label %eh.resume

lpad487:                                          ; preds = %invoke.cont459
  %225 = landingpad { i8*, i32 }
          cleanup
  %226 = extractvalue { i8*, i32 } %225, 0
  store i8* %226, i8** %exn.slot
  %227 = extractvalue { i8*, i32 } %225, 1
  store i32 %227, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call482) nounwind
  br label %eh.resume

lpad516:                                          ; preds = %invoke.cont488
  %228 = landingpad { i8*, i32 }
          cleanup
  %229 = extractvalue { i8*, i32 } %228, 0
  store i8* %229, i8** %exn.slot
  %230 = extractvalue { i8*, i32 } %228, 1
  store i32 %230, i32* %ehselector.slot
  call void @_ZdlPv(i8* %call511) nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad516, %lpad487, %lpad458, %lpad429, %lpad400, %lpad371, %lpad342, %lpad313, %lpad284, %lpad258, %invoke.cont92, %invoke.cont83, %invoke.cont74, %invoke.cont65, %invoke.cont56, %invoke.cont47, %invoke.cont38, %invoke.cont29, %invoke.cont20, %invoke.cont11, %invoke.cont4
  %exn = load i8*, i8** %exn.slot
  %sel = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val526 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val526

terminate.lpad:                                   ; preds = %lpad89, %lpad80, %lpad71, %lpad62, %lpad53, %lpad44, %lpad35, %lpad26, %lpad17, %lpad8, %lpad
  %231 = landingpad { i8*, i32 }
          catch i8* null
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable
}

declare void @_ZmlRKfRK9btVector3(%class.btVector3* noalias sret, float*, %class.btVector3*) inlinehint ssp

declare %class.btRigidBody* @_ZN7RagDoll20localCreateRigidBodyEfRK11btTransformP16btCollisionShape(%class.RagDoll*, float, %class.btTransform*, %class.btCollisionShape*) ssp align 2

declare void @_ZNK11btTransformmlERKS_(%class.btTransform* noalias sret, %class.btTransform*, %class.btTransform*) inlinehint ssp align 2

declare void @_ZN11btMatrix3x311setEulerZYXEfff(%class.btMatrix3x3*, float, float, float) ssp align 2

declare void @_ZN11btRigidBody10setDampingEff(%class.btRigidBody*, float, float)

declare void @_ZN17btCollisionObject19setDeactivationTimeEf(%class.btCollisionObject*, float) nounwind ssp align 2

declare void @_ZN11btRigidBody21setSleepingThresholdsEff(%class.btRigidBody*, float, float) nounwind ssp align 2

declare %class.btHingeConstraint* @_ZN17btHingeConstraintC1ER11btRigidBodyS1_RK11btTransformS4_b(%class.btHingeConstraint*, %class.btRigidBody*, %class.btRigidBody*, %class.btTransform*, %class.btTransform*, i1 zeroext)

declare void @_ZN17btHingeConstraint8setLimitEfffff(%class.btHingeConstraint*, float, float, float, float, float) ssp align 2

declare %class.btConeTwistConstraint* @_ZN21btConeTwistConstraintC1ER11btRigidBodyS1_RK11btTransformS4_(%class.btConeTwistConstraint*, %class.btRigidBody*, %class.btRigidBody*, %class.btTransform*, %class.btTransform*)

declare void @_ZN21btConeTwistConstraint8setLimitEffffff(%class.btConeTwistConstraint*, float, float, float, float, float, float) nounwind ssp align 2
