//===- maps.go - IR generation for maps -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for maps.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// makeMap implements make(maptype[, initial space])
func (fr *frame) makeMap(typ types.Type, cap_ *govalue) *govalue {
	// TODO(pcc): call __go_new_map_big here if needed
	dyntyp := fr.types.getMapDescriptorPointer(typ)
	dyntyp = fr.builder.CreateBitCast(dyntyp, llvm.PointerType(llvm.Int8Type(), 0), "")
	var cap llvm.Value
	if cap_ != nil {
		cap = fr.convert(cap_, types.Typ[types.Uintptr]).value
	} else {
		cap = llvm.ConstNull(fr.types.inttype)
	}
	m := fr.runtime.newMap.call(fr, dyntyp, cap)
	return newValue(m[0], typ)
}

// mapLookup implements v[, ok] = m[k]
func (fr *frame) mapLookup(m, k *govalue) (v *govalue, ok *govalue) {
	llk := k.value
	pk := fr.allocaBuilder.CreateAlloca(llk.Type(), "")
	fr.builder.CreateStore(llk, pk)
	valptr := fr.runtime.mapIndex.call(fr, m.value, pk, boolLLVMValue(false))[0]
	attrkind := llvm.AttributeKindID("nocapture")
	valptr.AddCallSiteAttribute(2, fr.types.ctx.CreateEnumAttribute(attrkind, 0))
	attrkind = llvm.AttributeKindID("readonly")
	valptr.AddCallSiteAttribute(2, fr.types.ctx.CreateEnumAttribute(attrkind, 0))
	okbit := fr.builder.CreateIsNotNull(valptr, "")

	elemtyp := m.Type().Underlying().(*types.Map).Elem()
	ok = newValue(fr.builder.CreateZExt(okbit, llvm.Int8Type(), ""), types.Typ[types.Bool])
	v = fr.loadOrNull(okbit, valptr, elemtyp)
	return
}

// mapUpdate implements m[k] = v
func (fr *frame) mapUpdate(m, k, v *govalue) {
	llk := k.value
	pk := fr.allocaBuilder.CreateAlloca(llk.Type(), "")
	fr.builder.CreateStore(llk, pk)
	valptr := fr.runtime.mapIndex.call(fr, m.value, pk, boolLLVMValue(true))[0]
	attrkind := llvm.AttributeKindID("nocapture")
	valptr.AddCallSiteAttribute(2, fr.types.ctx.CreateEnumAttribute(attrkind, 0))
	attrkind = llvm.AttributeKindID("readonly")
	valptr.AddCallSiteAttribute(2, fr.types.ctx.CreateEnumAttribute(attrkind, 0))

	elemtyp := m.Type().Underlying().(*types.Map).Elem()
	llelemtyp := fr.types.ToLLVM(elemtyp)
	typedvalptr := fr.builder.CreateBitCast(valptr, llvm.PointerType(llelemtyp, 0), "")
	fr.builder.CreateStore(v.value, typedvalptr)
}

// mapDelete implements delete(m, k)
func (fr *frame) mapDelete(m, k *govalue) {
	llk := k.value
	pk := fr.allocaBuilder.CreateAlloca(llk.Type(), "")
	fr.builder.CreateStore(llk, pk)
	fr.runtime.mapdelete.call(fr, m.value, pk)
}

// mapIterInit creates a map iterator
func (fr *frame) mapIterInit(m *govalue) []*govalue {
	// We represent an iterator as a tuple (map, *bool). The second element
	// controls whether the code we generate for "next" (below) calls the
	// runtime function for the first or the next element. We let the
	// optimizer reorganize this into something more sensible.
	isinit := fr.allocaBuilder.CreateAlloca(llvm.Int1Type(), "")
	fr.builder.CreateStore(llvm.ConstNull(llvm.Int1Type()), isinit)

	return []*govalue{m, newValue(isinit, types.NewPointer(types.Typ[types.Bool]))}
}

// mapIterNext advances the iterator, and returns the tuple (ok, k, v).
func (fr *frame) mapIterNext(iter []*govalue) []*govalue {
	maptyp := iter[0].Type().Underlying().(*types.Map)
	ktyp := maptyp.Key()
	klltyp := fr.types.ToLLVM(ktyp)
	vtyp := maptyp.Elem()
	vlltyp := fr.types.ToLLVM(vtyp)

	m, isinitptr := iter[0], iter[1]

	i8ptr := llvm.PointerType(llvm.Int8Type(), 0)
	mapiterbufty := llvm.ArrayType(i8ptr, 4)
	mapiterbuf := fr.allocaBuilder.CreateAlloca(mapiterbufty, "")
	mapiterbufelem0ptr := fr.builder.CreateStructGEP(mapiterbuf, 0, "")

	keybuf := fr.allocaBuilder.CreateAlloca(klltyp, "")
	keyptr := fr.builder.CreateBitCast(keybuf, i8ptr, "")
	valbuf := fr.allocaBuilder.CreateAlloca(vlltyp, "")
	valptr := fr.builder.CreateBitCast(valbuf, i8ptr, "")

	isinit := fr.builder.CreateLoad(isinitptr.value, "")

	initbb := llvm.AddBasicBlock(fr.function, "")
	nextbb := llvm.AddBasicBlock(fr.function, "")
	contbb := llvm.AddBasicBlock(fr.function, "")

	fr.builder.CreateCondBr(isinit, nextbb, initbb)

	fr.builder.SetInsertPointAtEnd(initbb)
	fr.builder.CreateStore(llvm.ConstAllOnes(llvm.Int1Type()), isinitptr.value)
	fr.runtime.mapiterinit.call(fr, m.value, mapiterbufelem0ptr)
	fr.builder.CreateBr(contbb)

	fr.builder.SetInsertPointAtEnd(nextbb)
	fr.runtime.mapiternext.call(fr, mapiterbufelem0ptr)
	fr.builder.CreateBr(contbb)

	fr.builder.SetInsertPointAtEnd(contbb)
	mapiterbufelem0 := fr.builder.CreateLoad(mapiterbufelem0ptr, "")
	okbit := fr.builder.CreateIsNotNull(mapiterbufelem0, "")
	ok := fr.builder.CreateZExt(okbit, llvm.Int8Type(), "")

	loadbb := llvm.AddBasicBlock(fr.function, "")
	cont2bb := llvm.AddBasicBlock(fr.function, "")
	fr.builder.CreateCondBr(okbit, loadbb, cont2bb)

	fr.builder.SetInsertPointAtEnd(loadbb)
	fr.runtime.mapiter2.call(fr, mapiterbufelem0ptr, keyptr, valptr)
	loadbb = fr.builder.GetInsertBlock()
	loadedkey := fr.builder.CreateLoad(keybuf, "")
	loadedval := fr.builder.CreateLoad(valbuf, "")
	fr.builder.CreateBr(cont2bb)

	fr.builder.SetInsertPointAtEnd(cont2bb)
	k := fr.builder.CreatePHI(klltyp, "")
	k.AddIncoming(
		[]llvm.Value{llvm.ConstNull(klltyp), loadedkey},
		[]llvm.BasicBlock{contbb, loadbb},
	)
	v := fr.builder.CreatePHI(vlltyp, "")
	v.AddIncoming(
		[]llvm.Value{llvm.ConstNull(vlltyp), loadedval},
		[]llvm.BasicBlock{contbb, loadbb},
	)

	return []*govalue{newValue(ok, types.Typ[types.Bool]), newValue(k, ktyp), newValue(v, vtyp)}
}
