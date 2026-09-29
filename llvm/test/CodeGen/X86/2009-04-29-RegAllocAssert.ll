; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -frame-pointer=all -relocation-model=pic
; PR4099

	%0 = type { [62 x ptr] }		; type %0
	%1 = type { ptr }		; type %1
	%2 = type { double }		; type %2
	%struct..5sPragmaType = type { ptr, i32 }
	%struct.AggInfo = type { i8, i8, i32, ptr, i32, ptr, i32, i32, i32, ptr, i32, i32 }
	%struct.AggInfo_col = type { ptr, i32, i32, i32, i32, ptr }
	%struct.AggInfo_func = type { ptr, ptr, i32, i32 }
	%struct.AuxData = type { ptr, ptr }
	%struct.Bitvec = type { i32, i32, i32, %0 }
	%struct.BtCursor = type { ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, %struct.CellInfo, i8, i8, ptr, i64, i32, i8, ptr }
	%struct.BtLock = type { ptr, i32, i8, ptr }
	%struct.BtShared = type { ptr, ptr, ptr, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i32, i16, i16, i32, i32, i32, i32, i8, i32, ptr, ptr, ptr, %struct.BusyHandler, i32, ptr, ptr, ptr }
	%struct.Btree = type { ptr, ptr, i8, i8, i8, i32, ptr, ptr }
	%struct.BtreeMutexArray = type { i32, [11 x ptr] }
	%struct.BusyHandler = type { ptr, ptr, i32 }
	%struct.CellInfo = type { ptr, i64, i32, i32, i16, i16, i16, i16 }
	%struct.CollSeq = type { ptr, i8, i8, ptr, ptr, ptr }
	%struct.Column = type { ptr, ptr, ptr, ptr, i8, i8, i8, i8 }
	%struct.Context = type { i64, i32, %struct.Fifo }
	%struct.CountCtx = type { i64 }
	%struct.Cursor = type { ptr, i32, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i64, ptr, i32, ptr, i64, ptr, ptr, i32, i64, ptr, ptr, i32, i32, ptr, ptr, ptr }
	%struct.Db = type { ptr, ptr, i8, i8, ptr, ptr, ptr }
	%struct.DbPage = type { ptr, i32, ptr, ptr, %struct.PagerLruLink, ptr, i8, i8, i8, i8, i8, i16, ptr, ptr, ptr }
	%struct.Expr = type { i8, i8, i16, ptr, ptr, ptr, ptr, %struct..5sPragmaType, %struct..5sPragmaType, i32, i32, ptr, i32, i32, ptr, ptr, i32 }
	%struct.ExprList = type { i32, i32, i32, ptr }
	%struct.ExprList_item = type { ptr, ptr, i8, i8, i8 }
	%struct.FKey = type { ptr, ptr, ptr, ptr, i32, ptr, i8, i8, i8, i8 }
	%struct.Fifo = type { i32, ptr, ptr }
	%struct.FifoPage = type { i32, i32, i32, ptr, [1 x i64] }
	%struct.FuncDef = type { i16, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, [1 x i8] }
	%struct.Hash = type { i8, i8, i32, i32, ptr, ptr }
	%struct.HashElem = type { ptr, ptr, ptr, ptr, i32 }
	%struct.IdList = type { ptr, i32, i32 }
	%struct.Index = type { ptr, i32, ptr, ptr, ptr, i32, i8, i8, ptr, ptr, ptr, ptr, ptr }
	%struct.KeyInfo = type { ptr, i8, i8, i8, i32, ptr, [1 x ptr] }
	%struct.Mem = type { %struct.CountCtx, double, ptr, ptr, i32, i16, i8, i8, ptr }
	%struct.MemPage = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i16, i16, i16, [5 x %struct._OvflCell], ptr, ptr, ptr, i32, ptr }
	%struct.Module = type { ptr, ptr, ptr, ptr }
	%struct.Op = type { i8, i8, i8, i8, i32, i32, i32, %1 }
	%struct.Pager = type { ptr, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.PagerLruList, ptr, ptr, ptr, i64, i64, i64, i64, i64, i32, ptr, ptr, i32, ptr, ptr, [16 x i8] }
	%struct.PagerLruLink = type { ptr, ptr }
	%struct.PagerLruList = type { ptr, ptr, ptr }
	%struct.Schema = type { i32, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Hash, ptr, i8, i8, i16, i32, ptr }
	%struct.Select = type { ptr, i8, i8, i8, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, [3 x i32] }
	%struct.SrcList = type { i16, i16, [1 x %struct.SrcList_item] }
	%struct.SrcList_item = type { ptr, ptr, ptr, ptr, ptr, i8, i8, i32, ptr, ptr, i64 }
	%struct.Table = type { ptr, i32, ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, i32, i8, i8, i8, i8, i8, i8, i8, ptr, ptr, i32, ptr, ptr }
	%struct.Trigger = type { ptr, ptr, i8, i8, ptr, ptr, %struct..5sPragmaType, ptr, ptr, ptr, ptr }
	%struct.TriggerStep = type { i32, i32, ptr, ptr, %struct..5sPragmaType, ptr, ptr, ptr, ptr, ptr }
	%struct.Vdbe = type { ptr, ptr, ptr, i32, i32, ptr, i32, i32, ptr, ptr, ptr, i32, ptr, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, %struct.Fifo, i32, i32, ptr, i32, i32, i32, i32, i32, [25 x i32], i32, i32, ptr, ptr, ptr, i8, i8, i8, i8, i8, i8, i32, i64, i32, %struct.BtreeMutexArray, i32, ptr, i32 }
	%struct.VdbeFunc = type { ptr, i32, [1 x %struct.AuxData] }
	%struct._OvflCell = type { ptr, i16 }
	%struct._ht = type { i32, ptr }
	%struct.sColMap = type { i32, ptr }
	%struct.sqlite3 = type { ptr, i32, ptr, i32, i32, i32, i32, i8, i8, i8, i8, i32, ptr, i64, i64, i32, i32, i32, ptr, %struct.sqlite3InitInfo, i32, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %2, ptr, ptr, ptr, ptr, i32, %struct.Hash, ptr, ptr, i32, %struct.Hash, %struct.Hash, %struct.BusyHandler, i32, [2 x %struct.Db], i8 }
	%struct.sqlite3InitInfo = type { i32, i32, i8 }
	%struct.sqlite3_context = type { ptr, ptr, %struct.Mem, ptr, i32, ptr }
	%struct.sqlite3_file = type { ptr }
	%struct.sqlite3_index_constraint = type { i32, i8, i8, i32 }
	%struct.sqlite3_index_constraint_usage = type { i32, i8 }
	%struct.sqlite3_index_info = type { i32, ptr, i32, ptr, ptr, i32, ptr, i32, i32, double }
	%struct.sqlite3_io_methods = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.sqlite3_module = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.sqlite3_mutex = type opaque
	%struct.sqlite3_vfs = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.sqlite3_vtab = type { ptr, i32, ptr }
	%struct.sqlite3_vtab_cursor = type { ptr }

define fastcc void @dropCell(ptr nocapture %pPage, i32 %idx, i32 %sz) nounwind ssp {
entry:
	%0 = load ptr, ptr null, align 8		; <ptr> [#uses=4]
	%1 = or i32 0, 0		; <i32> [#uses=1]
	%2 = icmp slt i32 %sz, 4		; <i1> [#uses=1]
	%size_addr.0.i = select i1 %2, i32 4, i32 %sz		; <i32> [#uses=1]
	br label %bb3.i

bb3.i:		; preds = %bb3.i, %entry
	%3 = icmp eq i32 0, 0		; <i1> [#uses=1]
	%or.cond.i = or i1 %3, false		; <i1> [#uses=1]
	br i1 %or.cond.i, label %bb5.i, label %bb3.i

bb5.i:		; preds = %bb3.i
	%4 = getelementptr i8, ptr %0, i64 0		; <ptr> [#uses=1]
	store i8 0, ptr %4, align 1
	%5 = getelementptr i8, ptr %0, i64 0		; <ptr> [#uses=1]
	store i8 0, ptr %5, align 1
	%6 = add i32 %1, 2		; <i32> [#uses=1]
	%7 = zext i32 %6 to i64		; <i64> [#uses=2]
	%8 = getelementptr i8, ptr %0, i64 %7		; <ptr> [#uses=1]
	%9 = lshr i32 %size_addr.0.i, 8		; <i32> [#uses=1]
	%10 = trunc i32 %9 to i8		; <i8> [#uses=1]
	store i8 %10, ptr %8, align 1
	%.sum31.i = add i64 %7, 1		; <i64> [#uses=1]
	%11 = getelementptr i8, ptr %0, i64 %.sum31.i		; <ptr> [#uses=1]
	store i8 0, ptr %11, align 1
	br label %bb11.outer.i

bb11.outer.i:		; preds = %bb11.outer.i, %bb5.i
	%12 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %12, label %bb12.i, label %bb11.outer.i

bb12.i:		; preds = %bb11.outer.i
	%i.08 = add i32 %idx, 1		; <i32> [#uses=1]
	%13 = icmp sgt i32 0, %i.08		; <i1> [#uses=1]
	br i1 %13, label %bb, label %bb2

bb:		; preds = %bb12.i
	br label %bb2

bb2:		; preds = %bb, %bb12.i
	%14 = getelementptr %struct.MemPage, ptr %pPage, i64 0, i32 1		; <ptr> [#uses=1]
	store i8 1, ptr %14, align 1
	ret void
}
