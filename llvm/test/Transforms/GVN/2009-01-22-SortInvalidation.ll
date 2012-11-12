; RUN: opt < %s -gvn | llvm-dis

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct..4sPragmaType = type { i8*, i32 }
	%struct.AggInfo = type { i8, i8, i32, %struct.ExprList*, i32, %struct.AggInfo_col*, i32, i32, i32, %struct.AggInfo_func*, i32, i32 }
	%struct.AggInfo_col = type { %struct.Table*, i32, i32, i32, i32, %struct.Expr* }
	%struct.AggInfo_func = type { %struct.Expr*, %struct.FuncDef*, i32, i32 }
	%struct.AuxData = type { i8*, void (i8*)* }
	%struct.Bitvec = type { i32, i32, i32, { [125 x i32] } }
	%struct.BtCursor = type { %struct.Btree*, %struct.BtShared*, %struct.BtCursor*, %struct.BtCursor*, i32 (i8*, i32, i8*, i32, i8*)*, i8*, i32, %struct.MemPage*, i32, %struct.CellInfo, i8, i8, i8*, i64, i32, i8, i32* }
	%struct.BtLock = type { %struct.Btree*, i32, i8, %struct.BtLock* }
	%struct.BtShared = type { %struct.Pager*, %struct.sqlite3*, %struct.BtCursor*, %struct.MemPage*, i8, i8, i8, i8, i8, i8, i8, i8, i32, i16, i16, i32, i32, i32, i32, i8, i32, i8*, void (i8*)*, %struct.sqlite3_mutex*, %struct.BusyHandler, i32, %struct.BtShared*, %struct.BtLock*, %struct.Btree* }
	%struct.Btree = type { %struct.sqlite3*, %struct.BtShared*, i8, i8, i8, i32, %struct.Btree*, %struct.Btree* }
	%struct.BtreeMutexArray = type { i32, [11 x %struct.Btree*] }
	%struct.BusyHandler = type { i32 (i8*, i32)*, i8*, i32 }
	%struct.CellInfo = type { i8*, i64, i32, i32, i16, i16, i16, i16 }
	%struct.CollSeq = type { i8*, i8, i8, i8*, i32 (i8*, i32, i8*, i32, i8*)*, void (i8*)* }
	%struct.Column = type { i8*, %struct.Expr*, i8*, i8*, i8, i8, i8, i8 }
	%struct.Context = type { i64, i32, %struct.Fifo }
	%struct.CountCtx = type { i64 }
	%struct.Cursor = type { %struct.BtCursor*, i32, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i64, %struct.Btree*, i32, i8*, i64, i8*, %struct.KeyInfo*, i32, i64, %struct.sqlite3_vtab_cursor*, %struct.sqlite3_module*, i32, i32, i32*, i32*, i8* }
	%struct.Db = type { i8*, %struct.Btree*, i8, i8, i8*, void (i8*)*, %struct.Schema* }
	%struct.Expr = type { i8, i8, i16, %struct.CollSeq*, %struct.Expr*, %struct.Expr*, %struct.ExprList*, %struct..4sPragmaType, %struct..4sPragmaType, i32, i32, %struct.AggInfo*, i32, i32, %struct.Select*, %struct.Table*, i32 }
	%struct.ExprList = type { i32, i32, i32, %struct.ExprList_item* }
	%struct.ExprList_item = type { %struct.Expr*, i8*, i8, i8, i8 }
	%struct.FKey = type { %struct.Table*, %struct.FKey*, i8*, %struct.FKey*, i32, %struct.sColMap*, i8, i8, i8, i8 }
	%struct.Fifo = type { i32, %struct.FifoPage*, %struct.FifoPage* }
	%struct.FifoPage = type { i32, i32, i32, %struct.FifoPage*, [1 x i64] }
	%struct.FuncDef = type { i16, i8, i8, i8, i8*, %struct.FuncDef*, void (%struct.sqlite3_context*, i32, %struct.Mem**)*, void (%struct.sqlite3_context*, i32, %struct.Mem**)*, void (%struct.sqlite3_context*)*, [1 x i8] }
	%struct.Hash = type { i8, i8, i32, i32, %struct.HashElem*, %struct._ht* }
	%struct.HashElem = type { %struct.HashElem*, %struct.HashElem*, i8*, i8*, i32 }
	%struct.IdList = type { %struct..4sPragmaType*, i32, i32 }
	%struct.Index = type { i8*, i32, i32*, i32*, %struct.Table*, i32, i8, i8, i8*, %struct.Index*, %struct.Schema*, i8*, i8** }
	%struct.KeyInfo = type { %struct.sqlite3*, i8, i8, i8, i32, i8*, [1 x %struct.CollSeq*] }
	%struct.Mem = type { %struct.CountCtx, double, %struct.sqlite3*, i8*, i32, i16, i8, i8, void (i8*)* }
	%struct.MemPage = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i16, i16, i16, [5 x %struct._OvflCell], %struct.BtShared*, i8*, %struct.PgHdr*, i32, %struct.MemPage* }
	%struct.Module = type { %struct.sqlite3_module*, i8*, i8*, void (i8*)* }
	%struct.Op = type { i8, i8, i8, i8, i32, i32, i32, { i32 } }
	%struct.Pager = type { %struct.sqlite3_vfs*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.Bitvec*, %struct.Bitvec*, i8*, i8*, i8*, i8*, %struct.sqlite3_file*, %struct.sqlite3_file*, %struct.sqlite3_file*, %struct.BusyHandler*, %struct.PagerLruList, %struct.PgHdr*, %struct.PgHdr*, %struct.PgHdr*, i64, i64, i64, i64, i64, i32, void (%struct.PgHdr*, i32)*, void (%struct.PgHdr*, i32)*, i32, %struct.PgHdr**, i8*, [16 x i8] }
	%struct.PagerLruLink = type { %struct.PgHdr*, %struct.PgHdr* }
	%struct.PagerLruList = type { %struct.PgHdr*, %struct.PgHdr*, %struct.PgHdr* }
	%struct.Parse = type { %struct.sqlite3*, i32, i8*, %struct.Vdbe*, i8, i8, i8, i8, i8, i8, i8, [8 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [12 x i32], i32, %struct.TableLock*, i32, i32, i32, i32, i32, %struct.Expr**, i8, %struct..4sPragmaType, %struct..4sPragmaType, %struct..4sPragmaType, i8*, i8*, %struct.Table*, %struct.Trigger*, %struct.TriggerStack*, i8*, %struct..4sPragmaType, i8, %struct.Table*, i32 }
	%struct.PgHdr = type { %struct.Pager*, i32, %struct.PgHdr*, %struct.PgHdr*, %struct.PagerLruLink, %struct.PgHdr*, i8, i8, i8, i8, i8, i16, %struct.PgHdr*, %struct.PgHdr*, i8* }
	%struct.Schema = type { i32, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Table*, i8, i8, i16, i32, %struct.sqlite3* }
	%struct.Select = type { %struct.ExprList*, i8, i8, i8, i8, i8, i8, i8, %struct.SrcList*, %struct.Expr*, %struct.ExprList*, %struct.Expr*, %struct.ExprList*, %struct.Select*, %struct.Select*, %struct.Select*, %struct.Expr*, %struct.Expr*, i32, i32, [3 x i32] }
	%struct.SrcList = type { i16, i16, [1 x %struct.SrcList_item] }
	%struct.SrcList_item = type { i8*, i8*, i8*, %struct.Table*, %struct.Select*, i8, i8, i32, %struct.Expr*, %struct.IdList*, i64 }
	%struct.Table = type { i8*, i32, %struct.Column*, i32, %struct.Index*, i32, %struct.Select*, i32, %struct.Trigger*, %struct.FKey*, i8*, %struct.Expr*, i32, i8, i8, i8, i8, i8, i8, i8, %struct.Module*, %struct.sqlite3_vtab*, i32, i8**, %struct.Schema* }
	%struct.TableLock = type { i32, i32, i8, i8* }
	%struct.Trigger = type { i8*, i8*, i8, i8, %struct.Expr*, %struct.IdList*, %struct..4sPragmaType, %struct.Schema*, %struct.Schema*, %struct.TriggerStep*, %struct.Trigger* }
	%struct.TriggerStack = type { %struct.Table*, i32, i32, i32, i32, i32, i32, %struct.Trigger*, %struct.TriggerStack* }
	%struct.TriggerStep = type { i32, i32, %struct.Trigger*, %struct.Select*, %struct..4sPragmaType, %struct.Expr*, %struct.ExprList*, %struct.IdList*, %struct.TriggerStep*, %struct.TriggerStep* }
	%struct.Vdbe = type { %struct.sqlite3*, %struct.Vdbe*, %struct.Vdbe*, i32, i32, %struct.Op*, i32, i32, i32*, %struct.Mem**, %struct.Mem*, i32, %struct.Cursor**, i32, %struct.Mem*, i8**, i32, i32, i32, %struct.Mem*, i32, i32, %struct.Fifo, i32, i32, %struct.Context*, i32, i32, i32, i32, i32, [25 x i32], i32, i32, i8**, i8*, %struct.Mem*, i8, i8, i8, i8, i8, i8, i32, i64, i32, %struct.BtreeMutexArray, i32, i8*, i32 }
	%struct.VdbeFunc = type { %struct.FuncDef*, i32, [1 x %struct.AuxData] }
	%struct._OvflCell = type { i8*, i16 }
	%struct._ht = type { i32, %struct.HashElem* }
	%struct.anon = type { double }
	%struct.sColMap = type { i32, i8* }
	%struct.sqlite3 = type { %struct.sqlite3_vfs*, i32, %struct.Db*, i32, i32, i32, i32, i8, i8, i8, i8, i32, %struct.CollSeq*, i64, i64, i32, i32, i32, %struct.sqlite3_mutex*, %struct.sqlite3InitInfo, i32, i8**, %struct.Vdbe*, i32, void (i8*, i8*)*, i8*, void (i8*, i8*, i64)*, i8*, i8*, i32 (i8*)*, i8*, void (i8*)*, i8*, void (i8*, i32, i8*, i8*, i64)*, void (i8*, %struct.sqlite3*, i32, i8*)*, void (i8*, %struct.sqlite3*, i32, i8*)*, i8*, %struct.Mem*, i8*, i8*, %struct.anon, i32 (i8*, i32, i8*, i8*, i8*, i8*)*, i8*, i32 (i8*)*, i8*, i32, %struct.Hash, %struct.Table*, %struct.sqlite3_vtab**, i32, %struct.Hash, %struct.Hash, %struct.BusyHandler, i32, [2 x %struct.Db], i8 }
	%struct.sqlite3InitInfo = type { i32, i32, i8 }
	%struct.sqlite3_context = type { %struct.FuncDef*, %struct.VdbeFunc*, %struct.Mem, %struct.Mem*, i32, %struct.CollSeq* }
	%struct.sqlite3_file = type { %struct.sqlite3_io_methods* }
	%struct.sqlite3_index_constraint = type { i32, i8, i8, i32 }
	%struct.sqlite3_index_constraint_usage = type { i32, i8 }
	%struct.sqlite3_index_info = type { i32, %struct.sqlite3_index_constraint*, i32, %struct.sqlite3_index_constraint_usage*, %struct.sqlite3_index_constraint_usage*, i32, i8*, i32, i32, double }
	%struct.sqlite3_io_methods = type { i32, i32 (%struct.sqlite3_file*)*, i32 (%struct.sqlite3_file*, i8*, i32, i64)*, i32 (%struct.sqlite3_file*, i8*, i32, i64)*, i32 (%struct.sqlite3_file*, i64)*, i32 (%struct.sqlite3_file*, i32)*, i32 (%struct.sqlite3_file*, i64*)*, i32 (%struct.sqlite3_file*, i32)*, i32 (%struct.sqlite3_file*, i32)*, i32 (%struct.sqlite3_file*)*, i32 (%struct.sqlite3_file*, i32, i8*)*, i32 (%struct.sqlite3_file*)*, i32 (%struct.sqlite3_file*)* }
	%struct.sqlite3_module = type { i32, i32 (%struct.sqlite3*, i8*, i32, i8**, %struct.sqlite3_vtab**, i8**)*, i32 (%struct.sqlite3*, i8*, i32, i8**, %struct.sqlite3_vtab**, i8**)*, i32 (%struct.sqlite3_vtab*, %struct.sqlite3_index_info*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*, %struct.sqlite3_vtab_cursor**)*, i32 (%struct.sqlite3_vtab_cursor*)*, i32 (%struct.sqlite3_vtab_cursor*, i32, i8*, i32, %struct.Mem**)*, i32 (%struct.sqlite3_vtab_cursor*)*, i32 (%struct.sqlite3_vtab_cursor*)*, i32 (%struct.sqlite3_vtab_cursor*, %struct.sqlite3_context*, i32)*, i32 (%struct.sqlite3_vtab_cursor*, i64*)*, i32 (%struct.sqlite3_vtab*, i32, %struct.Mem**, i64*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*, i32, i8*, void (%struct.sqlite3_context*, i32, %struct.Mem**)**, i8**)*, i32 (%struct.sqlite3_vtab*, i8*)* }
	%struct.sqlite3_mutex = type opaque
	%struct.sqlite3_vfs = type { i32, i32, i32, %struct.sqlite3_vfs*, i8*, i8*, i32 (%struct.sqlite3_vfs*, i8*, %struct.sqlite3_file*, i32, i32*)*, i32 (%struct.sqlite3_vfs*, i8*, i32)*, i32 (%struct.sqlite3_vfs*, i8*, i32)*, i32 (%struct.sqlite3_vfs*, i32, i8*)*, i32 (%struct.sqlite3_vfs*, i8*, i32, i8*)*, i8* (%struct.sqlite3_vfs*, i8*)*, void (%struct.sqlite3_vfs*, i32, i8*)*, i8* (%struct.sqlite3_vfs*, i8*, i8*)*, void (%struct.sqlite3_vfs*, i8*)*, i32 (%struct.sqlite3_vfs*, i32, i8*)*, i32 (%struct.sqlite3_vfs*, i32)*, i32 (%struct.sqlite3_vfs*, double*)* }
	%struct.sqlite3_vtab = type { %struct.sqlite3_module*, i32, i8* }
	%struct.sqlite3_vtab_cursor = type { %struct.sqlite3_vtab* }

define fastcc void @sqlite3Insert(%struct.Parse* %pParse, %struct.SrcList* %pTabList, %struct.ExprList* %pList, %struct.Select* %pSelect, %struct.IdList* %pColumn, i32 %onError) nounwind {
entry:
	br i1 false, label %bb54, label %bb69.loopexit

bb54:		; preds = %entry
	br label %bb69.loopexit

bb59:		; preds = %bb63.preheader
	%0 = load %struct..4sPragmaType** %3, align 4		; <%struct..4sPragmaType*> [#uses=0]
	br label %bb65

bb65:		; preds = %bb63.preheader, %bb59
	%1 = load %struct..4sPragmaType** %4, align 4		; <%struct..4sPragmaType*> [#uses=0]
	br i1 false, label %bb67, label %bb63.preheader

bb67:		; preds = %bb65
	%2 = getelementptr %struct.IdList* %pColumn, i32 0, i32 0		; <%struct..4sPragmaType**> [#uses=0]
	unreachable

bb69.loopexit:		; preds = %bb54, %entry
	%3 = getelementptr %struct.IdList* %pColumn, i32 0, i32 0		; <%struct..4sPragmaType**> [#uses=1]
	%4 = getelementptr %struct.IdList* %pColumn, i32 0, i32 0		; <%struct..4sPragmaType**> [#uses=1]
	br label %bb63.preheader

bb63.preheader:		; preds = %bb69.loopexit, %bb65
	br i1 false, label %bb59, label %bb65
}
