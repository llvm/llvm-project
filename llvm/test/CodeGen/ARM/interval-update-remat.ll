; RUN: llc -verify-regalloc < %s
; PR27275: When enabling remat for vreg defined by PHIs, make sure the update
; of the live range removes dead phi. Otherwise, we may end up with PHIs with
; incorrect operands and that will trigger assertions or verifier failures
; in later passes.

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios9.0.0"

%class.SOCKSClientSocketPoolTest_AsyncSOCKSConnectError_Test.1.226.276.1301.2326 = type { %class.MockTransportClientSocketPool.0.225.275.1300.2325, i32 }
%class.MockTransportClientSocketPool.0.225.275.1300.2325 = type { i8 }
%class.StaticSocketDataProvider.6.231.281.1306.2331 = type { i8, %struct.MockConnect.5.230.280.1305.2330 }
%struct.MockConnect.5.230.280.1305.2330 = type { %class.IPEndPoint.4.229.279.1304.2329 }
%class.IPEndPoint.4.229.279.1304.2329 = type { %class.IPAddress.3.228.278.1303.2328 }
%class.IPAddress.3.228.278.1303.2328 = type { %"class.(anonymous namespace)::vector.2.227.277.1302.2327" }
%"class.(anonymous namespace)::vector.2.227.277.1302.2327" = type { i8 }
%class.TestCompletionCallback.9.234.284.1309.2334 = type { %class.TestCompletionCallbackTemplate.8.233.283.1308.2333, i32 }
%class.TestCompletionCallbackTemplate.8.233.283.1308.2333 = type { i32 }
%class.AssertionResult.24.249.299.1324.2349 = type { i8, %class.scoped_ptr.23.248.298.1323.2348 }
%class.scoped_ptr.23.248.298.1323.2348 = type { ptr }
%class.Trans_NS___1_basic_string.18.243.293.1318.2343 = type { %class.Trans_NS___1___libcpp_compressed_pair_imp.17.242.292.1317.2342 }
%class.Trans_NS___1___libcpp_compressed_pair_imp.17.242.292.1317.2342 = type { %"struct.Trans_NS___1_basic_string<char, int, int>::__rep.16.241.291.1316.2341" }
%"struct.Trans_NS___1_basic_string<char, int, int>::__rep.16.241.291.1316.2341" = type { %"struct.Trans_NS___1_basic_string<char, int, int>::__long.15.240.290.1315.2340" }
%"struct.Trans_NS___1_basic_string<char, int, int>::__long.15.240.290.1315.2340" = type { i64, i32 }
%class.AssertHelper.10.235.285.1310.2335 = type { i8 }
%class.Message.13.238.288.1313.2338 = type { %class.scoped_ptr.0.12.237.287.1312.2337 }
%class.scoped_ptr.0.12.237.287.1312.2337 = type { ptr }
%"class.(anonymous namespace)::basic_stringstream.11.236.286.1311.2336" = type { i8 }
%class.scoped_refptr.19.244.294.1319.2344 = type { i8 }
%class.BoundNetLog.20.245.295.1320.2345 = type { i32 }
%struct.MockReadWrite.7.232.282.1307.2332 = type { i32 }
%"class.(anonymous namespace)::basic_iostream.22.247.297.1322.2347" = type { i8 }
%class.ClientSocketHandle.14.239.289.1314.2339 = type { i8 }
%"class.(anonymous namespace)::__vector_base.21.246.296.1321.2346" = type { i8 }

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define void @_ZN53SOCKSClientSocketPoolTest_AsyncSOCKSConnectError_Test6m_fn10Ev(ptr %this) align 2 {
entry:
  %socket_data = alloca %class.StaticSocketDataProvider.6.231.281.1306.2331, align 1
  %agg.tmp = alloca %struct.MockConnect.5.230.280.1305.2330, align 1
  %callback = alloca %class.TestCompletionCallback.9.234.284.1309.2334, align 4
  %gtest_ar = alloca %class.AssertionResult.24.249.299.1324.2349, align 4
  %temp.lvalue = alloca %class.AssertHelper.10.235.285.1310.2335, align 1
  %agg.tmp10 = alloca %class.Message.13.238.288.1313.2338, align 4
  %ref.tmp = alloca %class.Trans_NS___1_basic_string.18.243.293.1318.2343, align 4
  %agg.tmp16 = alloca %class.scoped_refptr.19.244.294.1319.2344, align 1
  %agg.tmp18 = alloca %class.BoundNetLog.20.245.295.1320.2345, align 4
  %call2 = call ptr @_ZN24StaticSocketDataProviderC1EP13MockReadWritejS1_j(ptr nonnull %socket_data, ptr undef, i32 1, ptr null, i32 0)
  %call3 = call ptr @_ZN11MockConnectC1Ev(ptr nonnull %agg.tmp)
  call void @_ZN24StaticSocketDataProvider5m_fn8E11MockConnect(ptr nonnull %socket_data, ptr nonnull %agg.tmp)
  %call5 = call ptr @_ZN22TestCompletionCallbackC1Ev(ptr nonnull %callback)
  %call6 = call i32 @_ZN29MockTransportClientSocketPool5m_fn9Ev(ptr %this)
  call void @_Z11CmpHelperEQPcS_xx(ptr nonnull sret(%class.AssertionResult.24.249.299.1324.2349) %gtest_ar, ptr @.str, ptr @.str, i64 0, i64 undef)
  %tmp = load i8, ptr undef, align 4
  %tobool.i = icmp eq i8 %tmp, 0
  br i1 %tobool.i, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  br i1 undef, label %_ZN15AssertionResult5m_fn6Ev.exit, label %cond.true.i

cond.true.i:                                      ; preds = %if.else
  %call4.i = call ptr @_ZN25Trans_NS___1_basic_stringIciiE5m_fn1Ev(ptr nonnull undef)
  br label %_ZN15AssertionResult5m_fn6Ev.exit

_ZN15AssertionResult5m_fn6Ev.exit:                ; preds = %cond.true.i, %if.else
  %cond.i = phi ptr [ %call4.i, %cond.true.i ], [ @.str, %if.else ]
  %call9 = call ptr @_ZN12AssertHelperC1EPKc(ptr nonnull %temp.lvalue, ptr %cond.i)
  %call11 = call ptr @_ZN7MessageC1Ev(ptr nonnull %agg.tmp10)
  call void @_ZN12AssertHelperaSE7Message(ptr nonnull %temp.lvalue, ptr nonnull %agg.tmp10)
  %call.i.i.i.i27 = call zeroext i1 @_Z6IsTruev()
  %brmerge = or i1 false, undef
  br i1 %brmerge, label %_ZN7MessageD1Ev.exit33, label %delete.notnull.i.i.i.i32

delete.notnull.i.i.i.i32:                         ; preds = %_ZN15AssertionResult5m_fn6Ev.exit
  %call.i.i.i.i.i.i31 = call ptr @_ZN12_GLOBAL__N_114basic_iostreamD2Ev(ptr undef)
  call void @_ZdlPv(ptr undef)
  br label %_ZN7MessageD1Ev.exit33

_ZN7MessageD1Ev.exit33:                           ; preds = %delete.notnull.i.i.i.i32, %_ZN15AssertionResult5m_fn6Ev.exit
  %call13 = call ptr @_ZN12AssertHelperD1Ev(ptr nonnull %temp.lvalue)
  br label %if.end

if.end:                                           ; preds = %_ZN7MessageD1Ev.exit33, %entry
  %message_.i.i = getelementptr inbounds %class.AssertionResult.24.249.299.1324.2349, ptr %gtest_ar, i32 0, i32 1
  %call.i.i.i = call ptr @_ZN10scoped_ptrI25Trans_NS___1_basic_stringIciiEED2Ev(ptr %message_.i.i)
  call void @llvm.memset.p0.i32(ptr align 4 null, i8 0, i32 12, i1 false)
  call void @_ZN25Trans_NS___1_basic_stringIciiE5m_fn2Ev(ptr nonnull %ref.tmp)
  call void @_Z19CreateSOCKSv5Paramsv(ptr nonnull sret(%class.scoped_refptr.19.244.294.1319.2344) %agg.tmp16)
  %callback_.i = getelementptr inbounds %class.TestCompletionCallback.9.234.284.1309.2334, ptr %callback, i32 0, i32 1
  %pool_ = getelementptr inbounds %class.SOCKSClientSocketPoolTest_AsyncSOCKSConnectError_Test.1.226.276.1301.2326, ptr %this, i32 0, i32 1
  store i32 0, ptr %agg.tmp18, align 4
  call void @_ZN18ClientSocketHandle5m_fn3IPiEEvRK25Trans_NS___1_basic_stringIciiE13scoped_refptr15RequestPriorityN16ClientSocketPool13RespectLimitsERiT_11BoundNetLog(ptr nonnull undef, ptr nonnull dereferenceable(12) %ref.tmp, ptr nonnull %agg.tmp16, i32 0, i32 1, ptr nonnull dereferenceable(4) %callback_.i, ptr %pool_, ptr nonnull %agg.tmp18)
  %call19 = call ptr @_ZN11BoundNetLogD1Ev(ptr nonnull %agg.tmp18)
  call void @_Z11CmpHelperEQPcS_xx(ptr nonnull sret(%class.AssertionResult.24.249.299.1324.2349) undef, ptr @.str, ptr @.str, i64 -1, i64 0)
  br i1 undef, label %if.then.i.i.i.i, label %_ZN7MessageD1Ev.exit

if.then.i.i.i.i:                                  ; preds = %if.end
  %tmp2 = load ptr, ptr undef, align 4
  br label %_ZN7MessageD1Ev.exit

_ZN7MessageD1Ev.exit:                             ; preds = %if.then.i.i.i.i, %if.end
  %connect_.i.i = getelementptr inbounds %class.StaticSocketDataProvider.6.231.281.1306.2331, ptr %socket_data, i32 0, i32 1
  %call.i.i.i.i.i.i.i.i.i.i = call ptr @_ZN12_GLOBAL__N_113__vector_baseD2Ev(ptr %connect_.i.i)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

declare ptr @_ZN24StaticSocketDataProviderC1EP13MockReadWritejS1_j(ptr returned, ptr, i32, ptr, i32) unnamed_addr

declare void @_ZN24StaticSocketDataProvider5m_fn8E11MockConnect(ptr, ptr)

declare ptr @_ZN11MockConnectC1Ev(ptr returned) unnamed_addr

declare ptr @_ZN22TestCompletionCallbackC1Ev(ptr returned) unnamed_addr

declare i32 @_ZN29MockTransportClientSocketPool5m_fn9Ev(ptr)

declare ptr @_ZN12AssertHelperC1EPKc(ptr returned, ptr) unnamed_addr

declare void @_ZN12AssertHelperaSE7Message(ptr, ptr)

declare ptr @_ZN7MessageC1Ev(ptr returned) unnamed_addr

declare ptr @_ZN12AssertHelperD1Ev(ptr returned) unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

declare void @_ZN18ClientSocketHandle5m_fn3IPiEEvRK25Trans_NS___1_basic_stringIciiE13scoped_refptr15RequestPriorityN16ClientSocketPool13RespectLimitsERiT_11BoundNetLog(ptr, ptr dereferenceable(12), ptr, i32, i32, ptr dereferenceable(4), ptr, ptr)

declare void @_Z19CreateSOCKSv5Paramsv(ptr sret(%class.scoped_refptr.19.244.294.1319.2344))

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) #0

declare ptr @_ZN11BoundNetLogD1Ev(ptr returned) unnamed_addr

declare ptr @_ZN13scoped_refptrD1Ev(ptr returned) unnamed_addr

declare ptr @_ZN12_GLOBAL__N_113__vector_baseD2Ev(ptr returned) unnamed_addr

declare ptr @_ZN25Trans_NS___1_basic_stringIciiE5m_fn1Ev(ptr)

declare zeroext i1 @_Z6IsTruev()

declare void @_ZdlPv(ptr)

declare ptr @_ZN12_GLOBAL__N_114basic_iostreamD2Ev(ptr returned) unnamed_addr

declare ptr @_ZN10scoped_ptrI25Trans_NS___1_basic_stringIciiEED2Ev(ptr readonly returned) unnamed_addr align 2

declare void @_Z11CmpHelperEQPcS_xx(ptr sret(%class.AssertionResult.24.249.299.1324.2349), ptr, ptr, i64, i64)

declare void @_ZN25Trans_NS___1_basic_stringIciiE5m_fn2Ev(ptr)

attributes #0 = { argmemonly nounwind }
