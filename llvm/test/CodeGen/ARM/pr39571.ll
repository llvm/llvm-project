; RUN: llc < %s -mtriple armv4t-unknown-linux-gnueabi -mattr=+strict-align

; Avoid crash from forwarding indexed-loads back to store.
%struct.anon = type { %struct.ma*, %struct.mb }
%struct.ma = type { i8 }
%struct.mb = type { i8, i8 }
%struct.anon.0 = type { %struct.anon.1 }
%struct.anon.1 = type { %struct.ds }
%struct.ds = type <{ i8, %union.ie }>
%union.ie = type { %struct.ib }
%struct.ib = type { i8, i8, i16 }

@a = common dso_local local_unnamed_addr global %struct.anon* null, align 4
@b = common dso_local local_unnamed_addr global %struct.anon.0 zeroinitializer, align 1

; Function Attrs: norecurse nounwind
define dso_local void @func() local_unnamed_addr {
entry:
  %0 = load %struct.anon*, %struct.anon** @a, align 4
  %ad = getelementptr inbounds %struct.anon, %struct.anon* %0, i32 0, i32 0
  %1 = load %struct.ma*, %struct.ma** %ad, align 4
  %c.sroa.0.0..sroa_idx = getelementptr inbounds %struct.ma, %struct.ma* %1, i32 0, i32 0
  %c.sroa.0.0.copyload = load i8, i8* %c.sroa.0.0..sroa_idx, align 1
  %cb = getelementptr inbounds %struct.anon, %struct.anon* %0, i32 0, i32 1
  %band = getelementptr inbounds %struct.anon, %struct.anon* %0, i32 0, i32 1, i32 1
  store i8 %c.sroa.0.0.copyload, i8* %band, align 4
  store i8 6, i8* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @b, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0), align 1
  store i8 2, i8* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @b, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1), align 1
  %2 = bitcast %struct.mb* %cb to i32*
  %3 = load i32, i32* bitcast (i8* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @b, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0) to i32*), align 1
  store i32 %3, i32* %2, align 1
  ret void
}
