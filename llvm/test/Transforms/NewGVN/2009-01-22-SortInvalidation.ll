; RUN: opt < %s -passes=newgvn | llvm-dis

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct..4sPragmaType = type { ptr, i32 }
	%struct.AggInfo = type { i8, i8, i32, ptr, i32, ptr, i32, i32, i32, ptr, i32, i32 }
	%struct.AggInfo_col = type { ptr, i32, i32, i32, i32, ptr }
	%struct.AggInfo_func = type { ptr, ptr, i32, i32 }
	%struct.AuxData = type { ptr, ptr }
	%struct.Bitvec = type { i32, i32, i32, { [125 x i32] } }
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
	%struct.Expr = type { i8, i8, i16, ptr, ptr, ptr, ptr, %struct..4sPragmaType, %struct..4sPragmaType, i32, i32, ptr, i32, i32, ptr, ptr, i32 }
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
	%struct.Op = type { i8, i8, i8, i8, i32, i32, i32, { i32 } }
	%struct.Pager = type { ptr, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.PagerLruList, ptr, ptr, ptr, i64, i64, i64, i64, i64, i32, ptr, ptr, i32, ptr, ptr, [16 x i8] }
	%struct.PagerLruLink = type { ptr, ptr }
	%struct.PagerLruList = type { ptr, ptr, ptr }
	%struct.Parse = type { ptr, i32, ptr, ptr, i8, i8, i8, i8, i8, i8, i8, [8 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [12 x i32], i32, ptr, i32, i32, i32, i32, i32, ptr, i8, %struct..4sPragmaType, %struct..4sPragmaType, %struct..4sPragmaType, ptr, ptr, ptr, ptr, ptr, ptr, %struct..4sPragmaType, i8, ptr, i32 }
	%struct.PgHdr = type { ptr, i32, ptr, ptr, %struct.PagerLruLink, ptr, i8, i8, i8, i8, i8, i16, ptr, ptr, ptr }
	%struct.Schema = type { i32, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Hash, ptr, i8, i8, i16, i32, ptr }
	%struct.Select = type { ptr, i8, i8, i8, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, [3 x i32] }
	%struct.SrcList = type { i16, i16, [1 x %struct.SrcList_item] }
	%struct.SrcList_item = type { ptr, ptr, ptr, ptr, ptr, i8, i8, i32, ptr, ptr, i64 }
	%struct.Table = type { ptr, i32, ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, i32, i8, i8, i8, i8, i8, i8, i8, ptr, ptr, i32, ptr, ptr }
	%struct.TableLock = type { i32, i32, i8, ptr }
	%struct.Trigger = type { ptr, ptr, i8, i8, ptr, ptr, %struct..4sPragmaType, ptr, ptr, ptr, ptr }
	%struct.TriggerStack = type { ptr, i32, i32, i32, i32, i32, i32, ptr, ptr }
	%struct.TriggerStep = type { i32, i32, ptr, ptr, %struct..4sPragmaType, ptr, ptr, ptr, ptr, ptr }
	%struct.Vdbe = type { ptr, ptr, ptr, i32, i32, ptr, i32, i32, ptr, ptr, ptr, i32, ptr, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, %struct.Fifo, i32, i32, ptr, i32, i32, i32, i32, i32, [25 x i32], i32, i32, ptr, ptr, ptr, i8, i8, i8, i8, i8, i8, i32, i64, i32, %struct.BtreeMutexArray, i32, ptr, i32 }
	%struct.VdbeFunc = type { ptr, i32, [1 x %struct.AuxData] }
	%struct._OvflCell = type { ptr, i16 }
	%struct._ht = type { i32, ptr }
	%struct.anon = type { double }
	%struct.sColMap = type { i32, ptr }
	%struct.sqlite3 = type { ptr, i32, ptr, i32, i32, i32, i32, i8, i8, i8, i8, i32, ptr, i64, i64, i32, i32, i32, ptr, %struct.sqlite3InitInfo, i32, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.anon, ptr, ptr, ptr, ptr, i32, %struct.Hash, ptr, ptr, i32, %struct.Hash, %struct.Hash, %struct.BusyHandler, i32, [2 x %struct.Db], i8 }
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

define fastcc void @sqlite3Insert(ptr %pParse, ptr %pTabList, ptr %pList, ptr %pSelect, ptr %pColumn, i32 %onError) nounwind {
entry:
	br i1 false, label %bb54, label %bb69.loopexit

bb54:		; preds = %entry
	br label %bb69.loopexit

bb59:		; preds = %bb63.preheader
	%0 = load ptr, ptr %3, align 4		; <ptr> [#uses=0]
	br label %bb65

bb65:		; preds = %bb63.preheader, %bb59
	%1 = load ptr, ptr %4, align 4		; <ptr> [#uses=0]
	br i1 false, label %bb67, label %bb63.preheader

bb67:		; preds = %bb65
	%2 = getelementptr %struct.IdList, ptr %pColumn, i32 0, i32 0		; <ptr> [#uses=0]
	unreachable

bb69.loopexit:		; preds = %bb54, %entry
	%3 = getelementptr %struct.IdList, ptr %pColumn, i32 0, i32 0		; <ptr> [#uses=1]
	%4 = getelementptr %struct.IdList, ptr %pColumn, i32 0, i32 0		; <ptr> [#uses=1]
	br label %bb63.preheader

bb63.preheader:		; preds = %bb69.loopexit, %bb65
	br i1 false, label %bb59, label %bb65
}
