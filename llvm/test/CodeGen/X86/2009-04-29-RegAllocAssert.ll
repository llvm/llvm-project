; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -frame-pointer=all -relocation-model=pic
; PR4099

	%0 = type { [62 x %struct.Bitvec*] }		; type %0
	%1 = type { i8* }		; type %1
	%2 = type { double }		; type %2
	%struct..5sPragmaType = type { i8*, i32 }
	%struct.AggInfo = type { i8, i8, i32, %struct.ExprList*, i32, %struct.AggInfo_col*, i32, i32, i32, %struct.AggInfo_func*, i32, i32 }
	%struct.AggInfo_col = type { %struct.Table*, i32, i32, i32, i32, %struct.Expr* }
	%struct.AggInfo_func = type { %struct.Expr*, %struct.FuncDef*, i32, i32 }
	%struct.AuxData = type { i8*, void (i8*)* }
	%struct.Bitvec = type { i32, i32, i32, %0 }
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
	%struct.DbPage = type { %struct.Pager*, i32, %struct.DbPage*, %struct.DbPage*, %struct.PagerLruLink, %struct.DbPage*, i8, i8, i8, i8, i8, i16, %struct.DbPage*, %struct.DbPage*, i8* }
	%struct.Expr = type { i8, i8, i16, %struct.CollSeq*, %struct.Expr*, %struct.Expr*, %struct.ExprList*, %struct..5sPragmaType, %struct..5sPragmaType, i32, i32, %struct.AggInfo*, i32, i32, %struct.Select*, %struct.Table*, i32 }
	%struct.ExprList = type { i32, i32, i32, %struct.ExprList_item* }
	%struct.ExprList_item = type { %struct.Expr*, i8*, i8, i8, i8 }
	%struct.FKey = type { %struct.Table*, %struct.FKey*, i8*, %struct.FKey*, i32, %struct.sColMap*, i8, i8, i8, i8 }
	%struct.Fifo = type { i32, %struct.FifoPage*, %struct.FifoPage* }
	%struct.FifoPage = type { i32, i32, i32, %struct.FifoPage*, [1 x i64] }
	%struct.FuncDef = type { i16, i8, i8, i8, i8*, %struct.FuncDef*, void (%struct.sqlite3_context*, i32, %struct.Mem**)*, void (%struct.sqlite3_context*, i32, %struct.Mem**)*, void (%struct.sqlite3_context*)*, [1 x i8] }
	%struct.Hash = type { i8, i8, i32, i32, %struct.HashElem*, %struct._ht* }
	%struct.HashElem = type { %struct.HashElem*, %struct.HashElem*, i8*, i8*, i32 }
	%struct.IdList = type { %struct..5sPragmaType*, i32, i32 }
	%struct.Index = type { i8*, i32, i32*, i32*, %struct.Table*, i32, i8, i8, i8*, %struct.Index*, %struct.Schema*, i8*, i8** }
	%struct.KeyInfo = type { %struct.sqlite3*, i8, i8, i8, i32, i8*, [1 x %struct.CollSeq*] }
	%struct.Mem = type { %struct.CountCtx, double, %struct.sqlite3*, i8*, i32, i16, i8, i8, void (i8*)* }
	%struct.MemPage = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i16, i16, i16, [5 x %struct._OvflCell], %struct.BtShared*, i8*, %struct.DbPage*, i32, %struct.MemPage* }
	%struct.Module = type { %struct.sqlite3_module*, i8*, i8*, void (i8*)* }
	%struct.Op = type { i8, i8, i8, i8, i32, i32, i32, %1 }
	%struct.Pager = type { %struct.sqlite3_vfs*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.Bitvec*, %struct.Bitvec*, i8*, i8*, i8*, i8*, %struct.sqlite3_file*, %struct.sqlite3_file*, %struct.sqlite3_file*, %struct.BusyHandler*, %struct.PagerLruList, %struct.DbPage*, %struct.DbPage*, %struct.DbPage*, i64, i64, i64, i64, i64, i32, void (%struct.DbPage*, i32)*, void (%struct.DbPage*, i32)*, i32, %struct.DbPage**, i8*, [16 x i8] }
	%struct.PagerLruLink = type { %struct.DbPage*, %struct.DbPage* }
	%struct.PagerLruList = type { %struct.DbPage*, %struct.DbPage*, %struct.DbPage* }
	%struct.Schema = type { i32, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Table*, i8, i8, i16, i32, %struct.sqlite3* }
	%struct.Select = type { %struct.ExprList*, i8, i8, i8, i8, i8, i8, i8, %struct.SrcList*, %struct.Expr*, %struct.ExprList*, %struct.Expr*, %struct.ExprList*, %struct.Select*, %struct.Select*, %struct.Select*, %struct.Expr*, %struct.Expr*, i32, i32, [3 x i32] }
	%struct.SrcList = type { i16, i16, [1 x %struct.SrcList_item] }
	%struct.SrcList_item = type { i8*, i8*, i8*, %struct.Table*, %struct.Select*, i8, i8, i32, %struct.Expr*, %struct.IdList*, i64 }
	%struct.Table = type { i8*, i32, %struct.Column*, i32, %struct.Index*, i32, %struct.Select*, i32, %struct.Trigger*, %struct.FKey*, i8*, %struct.Expr*, i32, i8, i8, i8, i8, i8, i8, i8, %struct.Module*, %struct.sqlite3_vtab*, i32, i8**, %struct.Schema* }
	%struct.Trigger = type { i8*, i8*, i8, i8, %struct.Expr*, %struct.IdList*, %struct..5sPragmaType, %struct.Schema*, %struct.Schema*, %struct.TriggerStep*, %struct.Trigger* }
	%struct.TriggerStep = type { i32, i32, %struct.Trigger*, %struct.Select*, %struct..5sPragmaType, %struct.Expr*, %struct.ExprList*, %struct.IdList*, %struct.TriggerStep*, %struct.TriggerStep* }
	%struct.Vdbe = type { %struct.sqlite3*, %struct.Vdbe*, %struct.Vdbe*, i32, i32, %struct.Op*, i32, i32, i32*, %struct.Mem**, %struct.Mem*, i32, %struct.Cursor**, i32, %struct.Mem*, i8**, i32, i32, i32, %struct.Mem*, i32, i32, %struct.Fifo, i32, i32, %struct.Context*, i32, i32, i32, i32, i32, [25 x i32], i32, i32, i8**, i8*, %struct.Mem*, i8, i8, i8, i8, i8, i8, i32, i64, i32, %struct.BtreeMutexArray, i32, i8*, i32 }
	%struct.VdbeFunc = type { %struct.FuncDef*, i32, [1 x %struct.AuxData] }
	%struct._OvflCell = type { i8*, i16 }
	%struct._ht = type { i32, %struct.HashElem* }
	%struct.sColMap = type { i32, i8* }
	%struct.sqlite3 = type { %struct.sqlite3_vfs*, i32, %struct.Db*, i32, i32, i32, i32, i8, i8, i8, i8, i32, %struct.CollSeq*, i64, i64, i32, i32, i32, %struct.sqlite3_mutex*, %struct.sqlite3InitInfo, i32, i8**, %struct.Vdbe*, i32, void (i8*, i8*)*, i8*, void (i8*, i8*, i64)*, i8*, i8*, i32 (i8*)*, i8*, void (i8*)*, i8*, void (i8*, i32, i8*, i8*, i64)*, void (i8*, %struct.sqlite3*, i32, i8*)*, void (i8*, %struct.sqlite3*, i32, i8*)*, i8*, %struct.Mem*, i8*, i8*, %2, i32 (i8*, i32, i8*, i8*, i8*, i8*)*, i8*, i32 (i8*)*, i8*, i32, %struct.Hash, %struct.Table*, %struct.sqlite3_vtab**, i32, %struct.Hash, %struct.Hash, %struct.BusyHandler, i32, [2 x %struct.Db], i8 }
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

define fastcc void @dropCell(%struct.MemPage* nocapture %pPage, i32 %idx, i32 %sz) nounwind ssp {
entry:
	%0 = load i8*, i8** null, align 8		; <i8*> [#uses=4]
	%1 = or i32 0, 0		; <i32> [#uses=1]
	%2 = icmp slt i32 %sz, 4		; <i1> [#uses=1]
	%size_addr.0.i = select i1 %2, i32 4, i32 %sz		; <i32> [#uses=1]
	br label %bb3.i

bb3.i:		; preds = %bb3.i, %entry
	%3 = icmp eq i32 0, 0		; <i1> [#uses=1]
	%or.cond.i = or i1 %3, false		; <i1> [#uses=1]
	br i1 %or.cond.i, label %bb5.i, label %bb3.i

bb5.i:		; preds = %bb3.i
	%4 = getelementptr i8, i8* %0, i64 0		; <i8*> [#uses=1]
	store i8 0, i8* %4, align 1
	%5 = getelementptr i8, i8* %0, i64 0		; <i8*> [#uses=1]
	store i8 0, i8* %5, align 1
	%6 = add i32 %1, 2		; <i32> [#uses=1]
	%7 = zext i32 %6 to i64		; <i64> [#uses=2]
	%8 = getelementptr i8, i8* %0, i64 %7		; <i8*> [#uses=1]
	%9 = lshr i32 %size_addr.0.i, 8		; <i32> [#uses=1]
	%10 = trunc i32 %9 to i8		; <i8> [#uses=1]
	store i8 %10, i8* %8, align 1
	%.sum31.i = add i64 %7, 1		; <i64> [#uses=1]
	%11 = getelementptr i8, i8* %0, i64 %.sum31.i		; <i8*> [#uses=1]
	store i8 0, i8* %11, align 1
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
	%14 = getelementptr %struct.MemPage, %struct.MemPage* %pPage, i64 0, i32 1		; <i8*> [#uses=1]
	store i8 1, i8* %14, align 1
	ret void
}
