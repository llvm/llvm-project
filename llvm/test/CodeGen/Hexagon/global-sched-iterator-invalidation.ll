; RUN: llc -march=hexagon -mcpu=hexagonv73 -O2 < %s -o /dev/null
;
; Check that the Hexagon Global Scheduler does not crash due to iterator
; invalidation. With _GLIBCXX_DEBUG enabled (LLVM_ENABLE_EXPENSIVE_CHECKS + libstdc++),
; the old code triggered debug-mode assertions in std::__debug::vector when
; successor/predecessor iterators were invalidated during CFG modifications
; in performPullUp / updatePredecessors / AnalyzeBBBranches.

target triple = "hexagon-unknown-linux-musl"

%"class.std::__1::basic_string" = type { %struct.anon }
%struct.anon = type { %"union.std::__1::basic_string<char>::__rep" }
%"union.std::__1::basic_string<char>::__rep" = type { %"struct.std::__1::basic_string<char>::__long" }
%"struct.std::__1::basic_string<char>::__long" = type { %struct.anon.1, i32, ptr }
%struct.anon.1 = type { i32 }

@.str = private unnamed_addr constant [8 x i8] c"--crash\00", align 1
@.str.1 = private unnamed_addr constant [26 x i8] c"LLVM_DISABLE_CRASH_REPORT\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"1\00", align 1
@.str.3 = private unnamed_addr constant [27 x i8] c"LLVM_DISABLE_SYMBOLIZATION\00", align 1
@environ = external local_unnamed_addr global ptr, align 4
@.str.4 = private unnamed_addr constant [13 x i8] c"basic_string\00", align 1
@_ZTISt12length_error = external constant ptr
@_ZTVSt12length_error = external unnamed_addr constant { [5 x ptr] }, align 4

define dso_local noundef range(i32 0, 2) i32 @main(i32 noundef %argc, ptr noundef %argv) local_unnamed_addr #0 {
entry:
  %ref.tmp = alloca %"class.std::__1::basic_string", align 4
  %result = alloca i32, align 4
  %pid = alloca i32, align 4
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %argv, i32 4
  %dec = add nsw i32 %argc, -1
  %cmp = icmp sgt i32 %argc, 1
  br i1 %cmp, label %land.rhs, label %if.end

land.rhs:
  call void @llvm.lifetime.start.p0(ptr nonnull %ref.tmp) #5
  %0 = load ptr, ptr %incdec.ptr, align 4, !tbaa !6
  %call.i.i.i.i = tail call noundef i32 @strlen(ptr noundef nonnull dereferenceable(1) %0) #5
  %cmp.i.i.i = icmp ugt i32 %call.i.i.i.i, -10
  br i1 %cmp.i.i.i, label %if.then.i.i.i, label %if.end.i.i.i

if.then.i.i.i:
  tail call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE20__throw_length_errorB8ne210108Ev() #6
  unreachable

if.end.i.i.i:
  %cmp.i.i.i.i = icmp ult i32 %call.i.i.i.i, 11
  br i1 %cmp.i.i.i.i, label %if.end8.i.i.i, label %if.end8.thread.i.i.i

if.end8.thread.i.i.i:
  %sub.i.i.i.i = or i32 %call.i.i.i.i, 7
  %add.i.i.i = add nuw i32 %sub.i.i.i.i, 1
  %call2.i.i.i.i.i.i = tail call noalias noundef nonnull ptr @_Znwj(i32 noundef %add.i.i.i) #7
  %__data_.i22.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 8
  store ptr %call2.i.i.i.i.i.i, ptr %__data_.i22.i.i.i, align 4, !tbaa !9
  %bf.set5.i.i.i.i = or disjoint i32 %add.i.i.i, 1
  store i32 %bf.set5.i.i.i.i, ptr %ref.tmp, align 4
  %__size_.i.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 4
  store i32 %call.i.i.i.i, ptr %__size_.i.i.i.i, align 4, !tbaa !9
  br label %if.then.i.i.i.i.i

if.end8.i.i.i:
  %conv.i.i.i.i = trunc nuw nsw i32 %call.i.i.i.i to i8
  %bf.value.i.i.i.i = shl nuw nsw i8 %conv.i.i.i.i, 1
  store i8 %bf.value.i.i.i.i, ptr %ref.tmp, align 4
  %__data_.i.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 1
  %cmp.not.i.i.i.i.i = icmp eq i32 %call.i.i.i.i, 0
  br i1 %cmp.not.i.i.i.i.i, label %ctor.exit, label %if.then.i.i.i.i.i

if.then.i.i.i.i.i:
  %__p.025.i.i.i = phi ptr [ %call2.i.i.i.i.i.i, %if.end8.thread.i.i.i ], [ %__data_.i.i.i.i, %if.end8.i.i.i ]
  call void @llvm.memmove.p0.p0.i32(ptr nonnull align 1 %__p.025.i.i.i, ptr nonnull align 1 %0, i32 %call.i.i.i.i, i1 false)
  br label %ctor.exit

ctor.exit:
  %__p.026.i.i.i = phi ptr [ %__data_.i.i.i.i, %if.end8.i.i.i ], [ %__p.025.i.i.i, %if.then.i.i.i.i.i ]
  %arrayidx.i.i.i = getelementptr inbounds nuw i8, ptr %__p.026.i.i.i, i32 %call.i.i.i.i
  store i8 0, ptr %arrayidx.i.i.i, align 1, !tbaa !9
  %bf.load.i.i = load i8, ptr %ref.tmp, align 4
  %tobool.i.i = trunc i8 %bf.load.i.i to i1
  %__size_.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 4
  %1 = load i32, ptr %__size_.i.i.i, align 4
  %bf.lshr.i.i.i = lshr i8 %bf.load.i.i, 1
  %conv.i.i.i = zext nneg i8 %bf.lshr.i.i.i to i32
  %cond.i.i = select i1 %tobool.i.i, i32 %1, i32 %conv.i.i.i
  %cmp.not.i = icmp eq i32 %cond.i.i, 7
  br i1 %cmp.not.i, label %compare.exit.i, label %cleanup.action

compare.exit.i:
  %__data_.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 8
  %2 = load ptr, ptr %__data_.i.i.i.i.i, align 4
  %__data_.i4.i.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 1
  %cond.i.i21.i.i = select i1 %tobool.i.i, ptr %2, ptr %__data_.i4.i.i.i.i
  %bcmp.i = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %cond.i.i21.i.i, ptr noundef nonnull dereferenceable(7) @.str, i32 7)
  %cmp7.i.i = icmp eq i32 %bcmp.i, 0
  br label %cleanup.action

cleanup.action:
  %.ph = phi i1 [ false, %ctor.exit ], [ %cmp7.i.i, %compare.exit.i ]
  br i1 %tobool.i.i, label %if.then.i.i52, label %cleanup.done4

if.then.i.i52:
  %__data_.i.i.i = getelementptr inbounds nuw i8, ptr %ref.tmp, i32 8
  %3 = load ptr, ptr %__data_.i.i.i, align 4, !tbaa !9
  %bf.load.i4.i.i = load i32, ptr %ref.tmp, align 4
  %bf.lshr.i.i.i53 = and i32 %bf.load.i4.i.i, -2
  tail call void @_ZdlPvj(ptr noundef %3, i32 noundef %bf.lshr.i.i.i53) #8
  br label %cleanup.done4

cleanup.done4:
  call void @llvm.lifetime.end.p0(ptr nonnull %ref.tmp) #5
  br i1 %.ph, label %if.then, label %if.end11

if.then:
  %incdec.ptr5 = getelementptr inbounds nuw i8, ptr %argv, i32 8
  %dec6 = add nsw i32 %argc, -2
  %call7 = tail call i32 @setenv(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.2, i32 noundef 0) #5
  %call8 = tail call i32 @setenv(ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.2, i32 noundef 0) #5
  br label %if.end

if.end:
  %argc.addr.0 = phi i32 [ %dec6, %if.then ], [ %dec, %entry ]
  %argv.addr.0 = phi ptr [ %incdec.ptr5, %if.then ], [ %incdec.ptr, %entry ]
  %cmp9 = icmp eq i32 %argc.addr.0, 0
  br i1 %cmp9, label %cleanup42, label %if.end11

if.end11:
  %expectCrash.0.off059 = phi i1 [ %cmp, %if.end ], [ false, %cleanup.done4 ]
  %argv.addr.058 = phi ptr [ %argv.addr.0, %if.end ], [ %incdec.ptr, %cleanup.done4 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %result) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %pid) #5
  %4 = load ptr, ptr %argv.addr.058, align 4, !tbaa !6
  %5 = load ptr, ptr @environ, align 4, !tbaa !10
  %call13 = call i32 @posix_spawn(ptr noundef nonnull %pid, ptr noundef %4, ptr noundef null, ptr noundef null, ptr noundef nonnull %argv.addr.058, ptr noundef %5) #5
  %tobool.not = icmp eq i32 %call13, 0
  br i1 %tobool.not, label %if.end15, label %cleanup40

if.end15:
  %6 = load i32, ptr %pid, align 4, !tbaa !2
  %call16 = call i32 @waitpid(i32 noundef %6, ptr noundef nonnull %result, i32 noundef 10) #5
  %cmp17 = icmp eq i32 %call16, -1
  br i1 %cmp17, label %cleanup40, label %if.end19

if.end19:
  %7 = load i32, ptr %result, align 4, !tbaa !2
  %and = and i32 %7, 127
  %tobool20.not = icmp eq i32 %and, 0
  %and24 = and i32 %7, 65535
  %8 = add nsw i32 %and24, -256
  %cmp25 = icmp ult i32 %8, -255
  %signal.0 = or i1 %tobool20.not, %cmp25
  br i1 %signal.0, label %if.end33, label %if.then30

if.then30:
  %not.expectCrash.0.off059 = xor i1 %expectCrash.0.off059, true
  br label %cleanup40

if.end33:
  %9 = and i32 %7, 65280
  %10 = icmp ne i32 %9, 0
  %retcode.0.not = and i1 %10, %tobool20.not
  %not.retcode.0.not = xor i1 %retcode.0.not, true
  %narrow = or i1 %expectCrash.0.off059, %not.retcode.0.not
  br label %cleanup40

cleanup40:
  %retval.1.shrunk = phi i1 [ true, %if.end15 ], [ true, %if.end11 ], [ %not.expectCrash.0.off059, %if.then30 ], [ %narrow, %if.end33 ]
  %retval.1 = zext i1 %retval.1.shrunk to i32
  call void @llvm.lifetime.end.p0(ptr nonnull %pid) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %result) #5
  br label %cleanup42

cleanup42:
  %retval.2 = phi i32 [ %retval.1, %cleanup40 ], [ 1, %if.end ]
  ret i32 %retval.2
}

declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
declare i32 @setenv(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #2
declare i32 @posix_spawn(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #2
declare i32 @waitpid(i32 noundef, ptr noundef, i32 noundef) local_unnamed_addr #2
declare void @_ZdlPvj(ptr noundef, i32 noundef) local_unnamed_addr #3
declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE20__throw_length_errorB8ne210108Ev() #4
declare noundef nonnull ptr @_Znwj(i32 noundef) local_unnamed_addr #3
declare void @llvm.memmove.p0.p0.i32(ptr writeonly captures(none), ptr readonly captures(none), i32, i1 immarg) #1
declare i32 @strlen(ptr noundef captures(none)) local_unnamed_addr #2
declare i32 @bcmp(ptr captures(none), ptr captures(none), i32) local_unnamed_addr #2

attributes #0 = { mustprogress norecurse nounwind "no-trapping-math"="true" "target-cpu"="hexagonv73" "target-features"="+v73,-long-calls" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "target-cpu"="hexagonv73" "target-features"="+v73,-long-calls" }
attributes #3 = { nobuiltin nounwind "no-trapping-math"="true" "target-cpu"="hexagonv73" "target-features"="+v73,-long-calls" }
attributes #4 = { mustprogress noreturn nounwind }
attributes #5 = { nounwind }
attributes #6 = { noreturn }
attributes #7 = { builtin nounwind allocsize(0) }
attributes #8 = { builtin nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 23.0.0git"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!4, !4, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"p2 omnipotent char", !12, i64 0}
!12 = !{!"any p2 pointer", !8, i64 0}
