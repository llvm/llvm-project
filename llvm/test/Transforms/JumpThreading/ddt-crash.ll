; RUN: opt < %s -jump-threading -disable-output

%struct.ham = type { i8, i8, i16, i32 }
%struct.zot = type { i32 (...)** }
%struct.quux.0 = type { %struct.wombat }
%struct.wombat = type { %struct.zot }

@global = external global %struct.ham*, align 8
@global.1 = external constant i8*

declare i32 @wombat.2()

define void @blam() {
bb:
  %tmp = load i32, i32* undef
  %tmp1 = icmp eq i32 %tmp, 0
  br i1 %tmp1, label %bb11, label %bb2

bb2:
  %tmp3 = tail call i32 @wombat.2()
  switch i32 %tmp3, label %bb4 [
    i32 0, label %bb5
    i32 1, label %bb7
    i32 2, label %bb7
    i32 3, label %bb11
  ]

bb4:
  br label %bb7

bb5:
  %tmp6 = tail call i32 @wombat.2()
  br label %bb7

bb7:
  %tmp8 = phi i32 [ 0, %bb5 ], [ 1, %bb4 ], [ 2, %bb2 ], [ 2, %bb2 ]
  %tmp9 = icmp eq i32 %tmp8, 0
  br i1 %tmp9, label %bb11, label %bb10

bb10:
  ret void

bb11:
  ret void
}

define void @spam(%struct.ham* %arg) {
bb:
  %tmp = load i8, i8* undef, align 8
  switch i8 %tmp, label %bb11 [
    i8 1, label %bb11
    i8 2, label %bb11
    i8 3, label %bb1
    i8 4, label %bb1
  ]

bb1:
  br label %bb2

bb2:
  %tmp3 = phi i32 [ 0, %bb1 ], [ %tmp3, %bb8 ]
  br label %bb4

bb4:
  %tmp5 = load i8, i8* undef, align 8
  switch i8 %tmp5, label %bb11 [
    i8 0, label %bb11
    i8 1, label %bb10
    i8 2, label %bb10
    i8 3, label %bb6
    i8 4, label %bb6
  ]

bb6:
  br label %bb7

bb7:
  br i1 undef, label %bb8, label %bb10

bb8:
  %tmp9 = icmp eq %struct.ham* undef, %arg
  br i1 %tmp9, label %bb10, label %bb2

bb10:
  switch i32 %tmp3, label %bb4 [
    i32 0, label %bb14
    i32 1, label %bb11
    i32 2, label %bb12
  ]

bb11:
  unreachable

bb12:
  %tmp13 = load %struct.ham*, %struct.ham** undef
  br label %bb14

bb14:
  %tmp15 = phi %struct.ham* [ %tmp13, %bb12 ], [ null, %bb10 ]
  br label %bb16

bb16:
  %tmp17 = load i8, i8* undef, align 8
  switch i8 %tmp17, label %bb11 [
    i8 0, label %bb11
    i8 11, label %bb18
    i8 12, label %bb18
  ]

bb18:
  br label %bb19

bb19:
  br label %bb20

bb20:
  %tmp21 = load %struct.ham*, %struct.ham** undef
  switch i8 undef, label %bb22 [
    i8 0, label %bb4
    i8 11, label %bb10
    i8 12, label %bb10
  ]

bb22:
  br label %bb23

bb23:
  %tmp24 = icmp eq %struct.ham* %tmp21, null
  br i1 %tmp24, label %bb35, label %bb25

bb25:
  %tmp26 = icmp eq %struct.ham* %tmp15, null
  br i1 %tmp26, label %bb34, label %bb27

bb27:
  %tmp28 = load %struct.ham*, %struct.ham** undef
  %tmp29 = icmp eq %struct.ham* %tmp28, %tmp21
  br i1 %tmp29, label %bb35, label %bb30

bb30:
  br label %bb31

bb31:
  %tmp32 = load i8, i8* undef, align 8
  %tmp33 = icmp eq i8 %tmp32, 0
  br i1 %tmp33, label %bb31, label %bb34

bb34:
  br label %bb35

bb35:
  %tmp36 = phi i1 [ true, %bb34 ], [ false, %bb23 ], [ true, %bb27 ]
  br label %bb37

bb37:
  %tmp38 = icmp eq %struct.ham* %tmp15, null
  br i1 %tmp38, label %bb39, label %bb41

bb39:
  %tmp40 = load %struct.ham*, %struct.ham** @global
  br label %bb41

bb41:
  %tmp42 = select i1 %tmp36, %struct.ham* undef, %struct.ham* undef
  ret void
}

declare i32 @foo(...)

define void @zot() align 2 personality i8* bitcast (i32 (...)* @foo to i8*) {
bb:
  invoke void @bar()
          to label %bb1 unwind label %bb3

bb1:
  invoke void @bar()
          to label %bb2 unwind label %bb4

bb2:
  invoke void @bar()
          to label %bb6 unwind label %bb17

bb3:
  %tmp = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  unreachable

bb4:
  %tmp5 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  unreachable

bb6:
  invoke void @bar()
          to label %bb7 unwind label %bb19

bb7:
  invoke void @bar()
          to label %bb10 unwind label %bb8

bb8:
  %tmp9 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  unreachable

bb10:
  %tmp11 = load i32 (%struct.zot*)*, i32 (%struct.zot*)** undef, align 8
  %tmp12 = invoke i32 %tmp11(%struct.zot* nonnull undef)
          to label %bb13 unwind label %bb21

bb13:
  invoke void @bar()
          to label %bb14 unwind label %bb23

bb14:
  %tmp15 = load i32 (%struct.zot*)*, i32 (%struct.zot*)** undef, align 8
  %tmp16 = invoke i32 %tmp15(%struct.zot* nonnull undef)
          to label %bb26 unwind label %bb23

bb17:
  %tmp18 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  unreachable

bb19:
  %tmp20 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  unreachable

bb21:
  %tmp22 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  unreachable

bb23:
  %tmp24 = phi %struct.quux.0* [ null, %bb26 ], [ null, %bb14 ], [ undef, %bb13 ]
  %tmp25 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @global.1 to i8*)
          catch i8* null
  br label %bb30

bb26:
  %tmp27 = load i32 (%struct.zot*)*, i32 (%struct.zot*)** undef, align 8
  %tmp28 = invoke i32 %tmp27(%struct.zot* nonnull undef)
          to label %bb29 unwind label %bb23

bb29:
  unreachable

bb30:
  %tmp31 = icmp eq %struct.quux.0* %tmp24, null
  br i1 %tmp31, label %bb32, label %bb29

bb32:
  unreachable
}

declare void @bar()
