; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin9

	%"struct.Adv5::Ekin<3>" = type <{ i8 }>
	%"struct.Adv5::X::Energyflux<3>" = type { double }
	%"struct.BinaryNode<OpAdd,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > >" = type { %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >" }
	%"struct.BinaryNode<OpAdd,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > >" = type { %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >" }
	%"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >" = type { %"struct.Adv5::X::Energyflux<3>", %"struct.BinaryNode<OpAdd,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > >" }
	%"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > >" = type { %"struct.Adv5::X::Energyflux<3>", %"struct.BinaryNode<OpAdd,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > >" }
	%"struct.Centering<3>" = type { i32, i32, %"struct.std::vector<Loc<3>,std::allocator<Loc<3> > >", %"struct.std::vector<Vector<3, double, Full>,std::allocator<Vector<3, double, Full> > >" }
	%"struct.ContextMapper<1>" = type { i32 (...)** }
	%"struct.DataBlockController<double>" = type { %"struct.RefBlockController<double>", %"struct.Adv5::Ekin<3>"*, i8, %"struct.SingleObservable<int>", i32 }
	%"struct.DataBlockPtr<double,false>" = type { %"struct.RefCountedBlockPtr<double,false,DataBlockController<double> >" }
	%"struct.Domain<1,DomainTraits<Interval<1> > >" = type { %"struct.DomainBase<DomainTraits<Interval<1> > >" }
	%"struct.Domain<1,DomainTraits<Loc<1> > >" = type { %"struct.DomainBase<DomainTraits<Loc<1> > >" }
	%"struct.Domain<1,DomainTraits<Range<1> > >" = type { %"struct.DomainBase<DomainTraits<Range<1> > >" }
	%"struct.Domain<3,DomainTraits<Interval<3> > >" = type { %"struct.DomainBase<DomainTraits<Interval<3> > >" }
	%"struct.Domain<3,DomainTraits<Loc<3> > >" = type { %"struct.DomainBase<DomainTraits<Loc<3> > >" }
	%"struct.Domain<3,DomainTraits<Range<3> > >" = type { %"struct.DomainBase<DomainTraits<Range<3> > >" }
	%"struct.DomainBase<DomainTraits<Interval<1> > >" = type { [2 x i32] }
	%"struct.DomainBase<DomainTraits<Interval<3> > >" = type { [3 x %"struct.WrapNoInit<Interval<1> >"] }
	%"struct.DomainBase<DomainTraits<Loc<1> > >" = type { i32 }
	%"struct.DomainBase<DomainTraits<Loc<3> > >" = type { [3 x %"struct.WrapNoInit<Loc<1> >"] }
	%"struct.DomainBase<DomainTraits<Range<1> > >" = type { [3 x i32] }
	%"struct.DomainBase<DomainTraits<Range<3> > >" = type { [3 x %"struct.WrapNoInit<Range<1> >"] }
	%"struct.DomainLayout<3>" = type { %"struct.Node<Interval<3>,Interval<3> >" }
	%"struct.DomainMap<Interval<1>,int>" = type { i32, %"struct.DomainMapNode<Interval<1>,int>"*, %"struct.DomainMapIterator<Interval<1>,int>" }
	%"struct.DomainMapIterator<Interval<1>,int>" = type { %"struct.DomainMapNode<Interval<1>,int>"*, %"struct.std::_List_const_iterator<Interval<3> >" }
	%"struct.DomainMapNode<Interval<1>,int>" = type { %"struct.Interval<1>", %"struct.DomainMapNode<Interval<1>,int>"*, %"struct.DomainMapNode<Interval<1>,int>"*, %"struct.DomainMapNode<Interval<1>,int>"*, %"struct.std::list<Interval<3>,std::allocator<Interval<3> > >" }
	%"struct.Engine<3,Zero<double>,ConstantFunction>" = type { %"struct.Adv5::Ekin<3>", %"struct.Interval<3>", [3 x i32] }
	%"struct.Engine<3,double,Brick>" = type { %"struct.Pooma::BrickBase<3>", %"struct.DataBlockPtr<double,false>", double* }
	%"struct.Engine<3,double,BrickView>" = type { %"struct.Pooma::BrickViewBase<3>", %"struct.DataBlockPtr<double,false>", double* }
	%"struct.Engine<3,double,ConstantFunction>" = type { double, %"struct.Interval<3>", [3 x i32] }
	%"struct.Engine<3,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > > > >" = type { %"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >", %"struct.Interval<3>" }
	%"struct.Engine<3,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >" = type { %"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > >", %"struct.Interval<3>" }
	%"struct.Engine<3,double,MultiPatch<GridTag, Remote<Brick> > >" = type { %"struct.ContextMapper<1>", %"struct.GridLayout<3>", %"struct.RefCountedBlockPtr<Engine<3, double, Remote<Brick> >,false,RefBlockController<Engine<3, double, Remote<Brick> > > >", i32* }
	%"struct.Engine<3,double,MultiPatchView<GridTag, Remote<Brick>, 3> >" = type { %"struct.GridLayoutView<3,3>", %"struct.Engine<3,double,MultiPatch<GridTag, Remote<Brick> > >" }
	%"struct.Engine<3,double,Remote<Brick> >" = type { %"struct.Interval<3>", i32, %"struct.RefCountedPtr<Shared<Engine<3, double, Brick> > >" }
	%"struct.Engine<3,double,Remote<BrickView> >" = type { %"struct.Interval<3>", i32, %"struct.RefCountedPtr<Shared<Engine<3, double, BrickView> > >" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,Zero<double>,ConstantFunction>" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,Zero<double>,ConstantFunction>" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ConstantFunction>" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ConstantFunction>" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > > > >" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > > > >" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >" }
	%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >" = type { %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >" }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,Zero<double>,ConstantFunction>" = type { i32, %"struct.Centering<3>", i32, %"struct.RefCountedBlockPtr<FieldEngineBaseData<3, Zero<double>, ConstantFunction>,false,RefBlockController<FieldEngineBaseData<3, Zero<double>, ConstantFunction> > >", %"struct.Interval<3>", %"struct.GuardLayers<3>", %"struct.UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ConstantFunction>" = type { i32, %"struct.Centering<3>", i32, %"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, ConstantFunction>,false,RefBlockController<FieldEngineBaseData<3, double, ConstantFunction> > >", %"struct.Interval<3>", %"struct.GuardLayers<3>", %"struct.UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > > > >" = type { %"struct.Engine<3,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > > > >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"* }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >" = type { %"struct.Engine<3,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >"* }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >" = type { i32, %"struct.Centering<3>", i32, %"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > >,false,RefBlockController<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > > > >", %"struct.Interval<3>", %"struct.GuardLayers<3>", %"struct.UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >" = type { i32, %"struct.Centering<3>", i32, %"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> >,false,RefBlockController<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >", %"struct.Interval<3>", %"struct.GuardLayers<3>", %"struct.UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" }
	%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >" = type { i32, %"struct.Centering<3>", i32, %"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, Remote<BrickView> >,false,RefBlockController<FieldEngineBaseData<3, double, Remote<BrickView> > > >", %"struct.Interval<3>", %"struct.GuardLayers<3>", %"struct.UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" }
	%"struct.FieldEngineBaseData<3,Zero<double>,ConstantFunction>" = type { %"struct.Engine<3,Zero<double>,ConstantFunction>", %struct.RelationList }
	%"struct.FieldEngineBaseData<3,double,ConstantFunction>" = type { %"struct.Engine<3,double,ConstantFunction>", %struct.RelationList }
	%"struct.FieldEngineBaseData<3,double,MultiPatch<GridTag, Remote<Brick> > >" = type { %"struct.Engine<3,double,MultiPatch<GridTag, Remote<Brick> > >", %struct.RelationList }
	%"struct.FieldEngineBaseData<3,double,MultiPatchView<GridTag, Remote<Brick>, 3> >" = type { %"struct.Engine<3,double,MultiPatchView<GridTag, Remote<Brick>, 3> >", %struct.RelationList }
	%"struct.FieldEngineBaseData<3,double,Remote<BrickView> >" = type { %"struct.Engine<3,double,Remote<BrickView> >", %struct.RelationList }
	%struct.GlobalIDDataBase = type { %"struct.std::vector<GlobalIDDataBase::Pack,std::allocator<GlobalIDDataBase::Pack> >", %"struct.std::map<int,InformStream*,std::less<int>,std::allocator<std::pair<const int, InformStream*> > >" }
	%"struct.GlobalIDDataBase::Pack" = type { i32, i32, i32, i32 }
	%"struct.GridLayout<3>" = type { %"struct.ContextMapper<1>", %"struct.LayoutBase<3,GridLayoutData<3> >", %"struct.Observable<GridLayout<3> >" }
	%"struct.GridLayoutData<3>" = type { %"struct.LayoutBaseData<3>", %struct.RefCounted, [21 x i8], i8, [3 x i32], [3 x %"struct.DomainMap<Interval<1>,int>"], [3 x %"struct.DomainMap<Interval<1>,int>"] }
	%"struct.GridLayoutView<3,3>" = type { %"struct.LayoutBaseView<3,3,GridLayoutViewData<3, 3> >" }
	%"struct.GridLayoutViewData<3,3>" = type { %"struct.LayoutBaseViewData<3,3,GridLayout<3> >", %struct.RefCounted }
	%"struct.GuardLayers<3>" = type { [3 x i32], [3 x i32] }
	%"struct.INode<3>" = type { %"struct.Interval<3>", %struct.GlobalIDDataBase*, i32 }
	%"struct.Interval<1>" = type { %"struct.Domain<1,DomainTraits<Interval<1> > >" }
	%"struct.Interval<3>" = type { %"struct.Domain<3,DomainTraits<Interval<3> > >" }
	%"struct.LayoutBase<3,GridLayoutData<3> >" = type { %"struct.RefCountedPtr<GridLayoutData<3> >" }
	%"struct.LayoutBaseData<3>" = type { i32, %"struct.Interval<3>", %"struct.Interval<3>", %"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >", %"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >", %"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >", i8, i8, %"struct.GuardLayers<3>", %"struct.GuardLayers<3>", %"struct.std::vector<LayoutBaseData<3>::GCFillInfo,std::allocator<LayoutBaseData<3>::GCFillInfo> >", [3 x i32], [3 x i32], %"struct.Loc<3>" }
	%"struct.LayoutBaseData<3>::GCFillInfo" = type { %"struct.Interval<3>", i32, i32, i32 }
	%"struct.LayoutBaseView<3,3,GridLayoutViewData<3, 3> >" = type { %"struct.RefCountedPtr<GridLayoutViewData<3, 3> >" }
	%"struct.LayoutBaseViewData<3,3,GridLayout<3> >" = type { i32, %"struct.GridLayout<3>", %"struct.GuardLayers<3>", %"struct.GuardLayers<3>", %"struct.ViewIndexer<3,3>", %"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >", %"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >", %"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >", i8 }
	%"struct.Loc<1>" = type { %"struct.Domain<1,DomainTraits<Loc<1> > >" }
	%"struct.Loc<3>" = type { %"struct.Domain<3,DomainTraits<Loc<3> > >" }
	%"struct.MultiArg6<Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, ConstantFunction>,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, Zero<double>, ConstantFunction> >" = type { %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ConstantFunction>", %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,Zero<double>,ConstantFunction>" }
	%"struct.NoMeshData<3>" = type { %struct.RefCounted, %"struct.Interval<3>", %"struct.Interval<3>", %"struct.Interval<3>", %"struct.Interval<3>" }
	%"struct.Node<Interval<3>,Interval<3> >" = type { %"struct.Interval<3>", %"struct.Interval<3>", i32, i32, i32, i32 }
	%"struct.Observable<GridLayout<3> >" = type { %"struct.GridLayout<3>"*, %"struct.std::vector<Observer<GridLayout<3> >*,std::allocator<Observer<GridLayout<3> >*> >", i32, %"struct.Adv5::Ekin<3>" }
	%"struct.Pooma::BrickBase<3>" = type { %"struct.DomainLayout<3>", [3 x i32], [3 x i32], i32, i8 }
	%"struct.Pooma::BrickViewBase<3>" = type { %"struct.Interval<3>", [3 x i32], [3 x i32], i32, i8 }
	%"struct.Range<1>" = type { %"struct.Domain<1,DomainTraits<Range<1> > >" }
	%"struct.Range<3>" = type { %"struct.Domain<3,DomainTraits<Range<3> > >" }
	%"struct.RefBlockController<Engine<3, double, Remote<Brick> > >" = type { %struct.RefCounted, %"struct.Engine<3,double,Remote<Brick> >"*, %"struct.Engine<3,double,Remote<Brick> >"*, %"struct.Engine<3,double,Remote<Brick> >"*, i8 }
	%"struct.RefBlockController<FieldEngineBaseData<3, Zero<double>, ConstantFunction> >" = type { %struct.RefCounted, %"struct.FieldEngineBaseData<3,Zero<double>,ConstantFunction>"*, %"struct.FieldEngineBaseData<3,Zero<double>,ConstantFunction>"*, %"struct.FieldEngineBaseData<3,Zero<double>,ConstantFunction>"*, i8 }
	%"struct.RefBlockController<FieldEngineBaseData<3, double, ConstantFunction> >" = type { %struct.RefCounted, %"struct.FieldEngineBaseData<3,double,ConstantFunction>"*, %"struct.FieldEngineBaseData<3,double,ConstantFunction>"*, %"struct.FieldEngineBaseData<3,double,ConstantFunction>"*, i8 }
	%"struct.RefBlockController<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > > >" = type { %struct.RefCounted, %"struct.FieldEngineBaseData<3,double,MultiPatch<GridTag, Remote<Brick> > >"*, %"struct.FieldEngineBaseData<3,double,MultiPatch<GridTag, Remote<Brick> > >"*, %"struct.FieldEngineBaseData<3,double,MultiPatch<GridTag, Remote<Brick> > >"*, i8 }
	%"struct.RefBlockController<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> > >" = type { %struct.RefCounted, %"struct.FieldEngineBaseData<3,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*, %"struct.FieldEngineBaseData<3,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*, %"struct.FieldEngineBaseData<3,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*, i8 }
	%"struct.RefBlockController<FieldEngineBaseData<3, double, Remote<BrickView> > >" = type { %struct.RefCounted, %"struct.FieldEngineBaseData<3,double,Remote<BrickView> >"*, %"struct.FieldEngineBaseData<3,double,Remote<BrickView> >"*, %"struct.FieldEngineBaseData<3,double,Remote<BrickView> >"*, i8 }
	%"struct.RefBlockController<double>" = type { %struct.RefCounted, double*, double*, double*, i8 }
	%struct.RefCounted = type { i32, %"struct.Adv5::Ekin<3>" }
	%"struct.RefCountedBlockPtr<Engine<3, double, Remote<Brick> >,false,RefBlockController<Engine<3, double, Remote<Brick> > > >" = type { i32, %"struct.RefCountedPtr<RefBlockController<Engine<3, double, Remote<Brick> > > >" }
	%"struct.RefCountedBlockPtr<FieldEngineBaseData<3, Zero<double>, ConstantFunction>,false,RefBlockController<FieldEngineBaseData<3, Zero<double>, ConstantFunction> > >" = type { i32, %"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, Zero<double>, ConstantFunction> > >" }
	%"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, ConstantFunction>,false,RefBlockController<FieldEngineBaseData<3, double, ConstantFunction> > >" = type { i32, %"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, ConstantFunction> > >" }
	%"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > >,false,RefBlockController<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > > > >" = type { i32, %"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > > > >" }
	%"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> >,false,RefBlockController<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >" = type { i32, %"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >" }
	%"struct.RefCountedBlockPtr<FieldEngineBaseData<3, double, Remote<BrickView> >,false,RefBlockController<FieldEngineBaseData<3, double, Remote<BrickView> > > >" = type { i32, %"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, Remote<BrickView> > > >" }
	%"struct.RefCountedBlockPtr<double,false,DataBlockController<double> >" = type { i32, %"struct.RefCountedPtr<DataBlockController<double> >" }
	%"struct.RefCountedPtr<DataBlockController<double> >" = type { %"struct.DataBlockController<double>"* }
	%"struct.RefCountedPtr<GridLayoutData<3> >" = type { %"struct.GridLayoutData<3>"* }
	%"struct.RefCountedPtr<GridLayoutViewData<3, 3> >" = type { %"struct.GridLayoutViewData<3,3>"* }
	%"struct.RefCountedPtr<RefBlockController<Engine<3, double, Remote<Brick> > > >" = type { %"struct.RefBlockController<Engine<3, double, Remote<Brick> > >"* }
	%"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, Zero<double>, ConstantFunction> > >" = type { %"struct.RefBlockController<FieldEngineBaseData<3, Zero<double>, ConstantFunction> >"* }
	%"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, ConstantFunction> > >" = type { %"struct.RefBlockController<FieldEngineBaseData<3, double, ConstantFunction> >"* }
	%"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > > > >" = type { %"struct.RefBlockController<FieldEngineBaseData<3, double, MultiPatch<GridTag, Remote<Brick> > > >"* }
	%"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >" = type { %"struct.RefBlockController<FieldEngineBaseData<3, double, MultiPatchView<GridTag, Remote<Brick>, 3> > >"* }
	%"struct.RefCountedPtr<RefBlockController<FieldEngineBaseData<3, double, Remote<BrickView> > > >" = type { %"struct.RefBlockController<FieldEngineBaseData<3, double, Remote<BrickView> > >"* }
	%"struct.RefCountedPtr<RelationListData>" = type { %struct.RelationListData* }
	%"struct.RefCountedPtr<Shared<Engine<3, double, Brick> > >" = type { %"struct.Shared<Engine<3, double, Brick> >"* }
	%"struct.RefCountedPtr<Shared<Engine<3, double, BrickView> > >" = type { %"struct.Shared<Engine<3, double, BrickView> >"* }
	%"struct.RefCountedPtr<UniformRectilinearMeshData<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> > >" = type { %"struct.UniformRectilinearMeshData<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >"* }
	%struct.RelationList = type { %"struct.RefCountedPtr<RelationListData>" }
	%struct.RelationListData = type { %struct.RefCounted, %"struct.std::vector<RelationListItem*,std::allocator<RelationListItem*> >" }
	%struct.RelationListItem = type { i32 (...)**, i32, i32, i8 }
	%"struct.Shared<Engine<3, double, Brick> >" = type { %struct.RefCounted, %"struct.Engine<3,double,Brick>" }
	%"struct.Shared<Engine<3, double, BrickView> >" = type { %struct.RefCounted, %"struct.Engine<3,double,BrickView>" }
	%"struct.SingleObservable<int>" = type { %"struct.ContextMapper<1>"* }
	%"struct.UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" = type { %"struct.RefCountedPtr<UniformRectilinearMeshData<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> > >" }
	%"struct.UniformRectilinearMeshData<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >" = type { %"struct.NoMeshData<3>", %"struct.Vector<3,double,Full>", %"struct.Vector<3,double,Full>" }
	%"struct.Vector<3,double,Full>" = type { %"struct.VectorEngine<3,double,Full>" }
	%"struct.VectorEngine<3,double,Full>" = type { [3 x double] }
	%"struct.ViewIndexer<3,3>" = type { %"struct.Interval<3>", %"struct.Range<3>", [3 x i32], [3 x i32], %"struct.Loc<3>" }
	%"struct.WrapNoInit<Interval<1> >" = type { %"struct.Interval<1>" }
	%"struct.WrapNoInit<Loc<1> >" = type { %"struct.Loc<1>" }
	%"struct.WrapNoInit<Range<1> >" = type { %"struct.Range<1>" }
	%"struct.std::_List_base<Interval<3>,std::allocator<Interval<3> > >" = type { %"struct.std::_List_base<Interval<3>,std::allocator<Interval<3> > >::_List_impl" }
	%"struct.std::_List_base<Interval<3>,std::allocator<Interval<3> > >::_List_impl" = type { %"struct.std::_List_node_base" }
	%"struct.std::_List_const_iterator<Interval<3> >" = type { %"struct.std::_List_node_base"* }
	%"struct.std::_List_node_base" = type { %"struct.std::_List_node_base"*, %"struct.std::_List_node_base"* }
	%"struct.std::_Rb_tree<int,std::pair<const int, InformStream*>,std::_Select1st<std::pair<const int, InformStream*> >,std::less<int>,std::allocator<std::pair<const int, InformStream*> > >" = type { %"struct.std::_Rb_tree<int,std::pair<const int, InformStream*>,std::_Select1st<std::pair<const int, InformStream*> >,std::less<int>,std::allocator<std::pair<const int, InformStream*> > >::_Rb_tree_impl<std::less<int>,false>" }
	%"struct.std::_Rb_tree<int,std::pair<const int, InformStream*>,std::_Select1st<std::pair<const int, InformStream*> >,std::less<int>,std::allocator<std::pair<const int, InformStream*> > >::_Rb_tree_impl<std::less<int>,false>" = type { %"struct.Adv5::Ekin<3>", %"struct.std::_Rb_tree_node_base", i32 }
	%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
	%"struct.std::_Vector_base<GlobalIDDataBase::Pack,std::allocator<GlobalIDDataBase::Pack> >" = type { %"struct.std::_Vector_base<GlobalIDDataBase::Pack,std::allocator<GlobalIDDataBase::Pack> >::_Vector_impl" }
	%"struct.std::_Vector_base<GlobalIDDataBase::Pack,std::allocator<GlobalIDDataBase::Pack> >::_Vector_impl" = type { %"struct.GlobalIDDataBase::Pack"*, %"struct.GlobalIDDataBase::Pack"*, %"struct.GlobalIDDataBase::Pack"* }
	%"struct.std::_Vector_base<LayoutBaseData<3>::GCFillInfo,std::allocator<LayoutBaseData<3>::GCFillInfo> >" = type { %"struct.std::_Vector_base<LayoutBaseData<3>::GCFillInfo,std::allocator<LayoutBaseData<3>::GCFillInfo> >::_Vector_impl" }
	%"struct.std::_Vector_base<LayoutBaseData<3>::GCFillInfo,std::allocator<LayoutBaseData<3>::GCFillInfo> >::_Vector_impl" = type { %"struct.LayoutBaseData<3>::GCFillInfo"*, %"struct.LayoutBaseData<3>::GCFillInfo"*, %"struct.LayoutBaseData<3>::GCFillInfo"* }
	%"struct.std::_Vector_base<Loc<3>,std::allocator<Loc<3> > >" = type { %"struct.std::_Vector_base<Loc<3>,std::allocator<Loc<3> > >::_Vector_impl" }
	%"struct.std::_Vector_base<Loc<3>,std::allocator<Loc<3> > >::_Vector_impl" = type { %"struct.Loc<3>"*, %"struct.Loc<3>"*, %"struct.Loc<3>"* }
	%"struct.std::_Vector_base<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >" = type { %"struct.std::_Vector_base<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >::_Vector_impl" }
	%"struct.std::_Vector_base<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >::_Vector_impl" = type { %"struct.Node<Interval<3>,Interval<3> >"**, %"struct.Node<Interval<3>,Interval<3> >"**, %"struct.Node<Interval<3>,Interval<3> >"** }
	%"struct.std::_Vector_base<Observer<GridLayout<3> >*,std::allocator<Observer<GridLayout<3> >*> >" = type { %"struct.std::_Vector_base<Observer<GridLayout<3> >*,std::allocator<Observer<GridLayout<3> >*> >::_Vector_impl" }
	%"struct.std::_Vector_base<Observer<GridLayout<3> >*,std::allocator<Observer<GridLayout<3> >*> >::_Vector_impl" = type { %"struct.ContextMapper<1>"**, %"struct.ContextMapper<1>"**, %"struct.ContextMapper<1>"** }
	%"struct.std::_Vector_base<RelationListItem*,std::allocator<RelationListItem*> >" = type { %"struct.std::_Vector_base<RelationListItem*,std::allocator<RelationListItem*> >::_Vector_impl" }
	%"struct.std::_Vector_base<RelationListItem*,std::allocator<RelationListItem*> >::_Vector_impl" = type { %struct.RelationListItem**, %struct.RelationListItem**, %struct.RelationListItem** }
	%"struct.std::_Vector_base<Vector<3, double, Full>,std::allocator<Vector<3, double, Full> > >" = type { %"struct.std::_Vector_base<Vector<3, double, Full>,std::allocator<Vector<3, double, Full> > >::_Vector_impl" }
	%"struct.std::_Vector_base<Vector<3, double, Full>,std::allocator<Vector<3, double, Full> > >::_Vector_impl" = type { %"struct.Vector<3,double,Full>"*, %"struct.Vector<3,double,Full>"*, %"struct.Vector<3,double,Full>"* }
	%"struct.std::list<Interval<3>,std::allocator<Interval<3> > >" = type { %"struct.std::_List_base<Interval<3>,std::allocator<Interval<3> > >" }
	%"struct.std::map<int,InformStream*,std::less<int>,std::allocator<std::pair<const int, InformStream*> > >" = type { %"struct.std::_Rb_tree<int,std::pair<const int, InformStream*>,std::_Select1st<std::pair<const int, InformStream*> >,std::less<int>,std::allocator<std::pair<const int, InformStream*> > >" }
	%"struct.std::vector<GlobalIDDataBase::Pack,std::allocator<GlobalIDDataBase::Pack> >" = type { %"struct.std::_Vector_base<GlobalIDDataBase::Pack,std::allocator<GlobalIDDataBase::Pack> >" }
	%"struct.std::vector<LayoutBaseData<3>::GCFillInfo,std::allocator<LayoutBaseData<3>::GCFillInfo> >" = type { %"struct.std::_Vector_base<LayoutBaseData<3>::GCFillInfo,std::allocator<LayoutBaseData<3>::GCFillInfo> >" }
	%"struct.std::vector<Loc<3>,std::allocator<Loc<3> > >" = type { %"struct.std::_Vector_base<Loc<3>,std::allocator<Loc<3> > >" }
	%"struct.std::vector<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >" = type { %"struct.std::_Vector_base<Node<Interval<3>, Interval<3> >*,std::allocator<Node<Interval<3>, Interval<3> >*> >" }
	%"struct.std::vector<Observer<GridLayout<3> >*,std::allocator<Observer<GridLayout<3> >*> >" = type { %"struct.std::_Vector_base<Observer<GridLayout<3> >*,std::allocator<Observer<GridLayout<3> >*> >" }
	%"struct.std::vector<RelationListItem*,std::allocator<RelationListItem*> >" = type { %"struct.std::_Vector_base<RelationListItem*,std::allocator<RelationListItem*> >" }
	%"struct.std::vector<Vector<3, double, Full>,std::allocator<Vector<3, double, Full> > >" = type { %"struct.std::_Vector_base<Vector<3, double, Full>,std::allocator<Vector<3, double, Full> > >" }

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind

declare fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEEC1ERKSC_(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*) nounwind

declare fastcc void @_ZN9CenteringILi3EEC1ERKS0_i(%"struct.Centering<3>"*, %"struct.Centering<3>"*, i32) nounwind

declare fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEED1Ev(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >"*) nounwind

declare fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEEC1Id14MultiPatchViewI7GridTagS6_I5BrickELi3EEEERKS_IS5_T_T0_ERK5INodeILi3EE(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >"*, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*, %"struct.INode<3>"*) nounwind

declare fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EEED1Ev(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*) nounwind

declare fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EEEC1Id10MultiPatchIS7_SA_EEERKS_IS5_T_T0_ERK8IntervalILi3EE(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*, %"struct.Interval<3>"*) nounwind

define fastcc void @t(double %dt, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %rh, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %T, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %v, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %pg, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %ph, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %cs, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ConstantFunction>"* %cv, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,Zero<double>,ConstantFunction>"* %dlmdlt, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ConstantFunction>"* %xmue, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %vint, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %cent, %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %fvis, double %c_nr, double %c_av, i8 zeroext %cartvis_f) nounwind {
entry:
	%0 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >"*> [#uses=4]
	%s.i.i.i.i.i = alloca %"struct.Interval<3>"		; <%"struct.Interval<3>"*> [#uses=0]
	%1 = alloca %"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >"		; <%"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >"*> [#uses=2]
	%multiArg.i = alloca %"struct.MultiArg6<Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, ConstantFunction>,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, Zero<double>, ConstantFunction> >"		; <%"struct.MultiArg6<Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatch<GridTag, Remote<Brick> > >,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, ConstantFunction>,Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, Zero<double>, ConstantFunction> >"*> [#uses=0]
	%2 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=6]
	%3 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=2]
	%4 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=0]
	%5 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=2]
	%6 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=0]
	%7 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%8 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%9 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%10 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%11 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%12 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%13 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%14 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%15 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%16 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%17 = alloca %"struct.Interval<3>"		; <%"struct.Interval<3>"*> [#uses=0]
	%18 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%19 = alloca double		; <double*> [#uses=0]
	%20 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=0]
	%21 = alloca %"struct.Interval<3>"		; <%"struct.Interval<3>"*> [#uses=0]
	%22 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=0]
	%23 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=0]
	%24 = alloca %"struct.Interval<3>"		; <%"struct.Interval<3>"*> [#uses=0]
	%25 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%26 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%27 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%28 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%29 = alloca %"struct.Interval<3>"		; <%"struct.Interval<3>"*> [#uses=0]
	%30 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%31 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%32 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%33 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%34 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%35 = alloca %"struct.Interval<3>"		; <%"struct.Interval<3>"*> [#uses=0]
	%36 = alloca %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"		; <%"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=0]
	%37 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %v, i32 0, i32 0, i32 5		; <%"struct.GuardLayers<3>"*> [#uses=1]
	%38 = bitcast %"struct.GuardLayers<3>"* %37 to i8*		; <i8*> [#uses=1]
	br label %bb.i.i.i.i.i

bb.i.i.i.i.i:		; preds = %bb.i.i.i.i.i, %entry
	%39 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %39, label %_ZN14ScalarCodeInfoILi3ELi4EEC1Ev.exit.i, label %bb.i.i.i.i.i

_ZN14ScalarCodeInfoILi3ELi4EEC1Ev.exit.i:		; preds = %bb.i.i.i.i.i
	br label %bb.i.i.i35.i.i34

bb.i.i.i35.i.i34:		; preds = %bb.i.i.i35.i.i34, %_ZN14ScalarCodeInfoILi3ELi4EEC1Ev.exit.i
	%40 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %40, label %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit36.i.i37, label %bb.i.i.i35.i.i34

_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit36.i.i37:		; preds = %bb.i.i.i35.i.i34
	br label %bb.i.i.i19.i.i47

bb.i.i.i19.i.i47:		; preds = %bb.i.i.i19.i.i47, %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit36.i.i37
	%41 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %41, label %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i50, label %bb.i.i.i19.i.i47

_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i50:		; preds = %bb.i.i.i19.i.i47
	%42 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %rh, i32 0, i32 0		; <%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"*> [#uses=1]
	br label %bb.i.i.i19.i.i.i

bb.i.i.i19.i.i.i:		; preds = %bb.i.i.i19.i.i.i, %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i50
	%43 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %43, label %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i.i, label %bb.i.i.i19.i.i.i

_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i.i:		; preds = %bb.i.i.i19.i.i.i
	br label %bb.i.i.i35.i.i433

bb.i.i.i35.i.i433:		; preds = %bb.i.i.i35.i.i433, %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i.i
	%44 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %44, label %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit36.i.i436, label %bb.i.i.i35.i.i433

_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit36.i.i436:		; preds = %bb.i.i.i35.i.i433
	br label %bb.i.i.i19.i.i446

bb.i.i.i19.i.i446:		; preds = %bb.i.i.i19.i.i446, %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit36.i.i436
	%45 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %45, label %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i449, label %bb.i.i.i19.i.i446

_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i449:		; preds = %bb.i.i.i19.i.i446
	br label %bb.i.i.i.i.i459

bb.i.i.i.i.i459:		; preds = %bb.i.i.i.i.i459, %_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_.exit20.i.i449
	%46 = icmp eq i32* null, null		; <i1> [#uses=1]
	br i1 %46, label %_ZN14ScalarCodeInfoILi3ELi6EEC1Ev.exit.i460, label %bb.i.i.i.i.i459

_ZN14ScalarCodeInfoILi3ELi6EEC1Ev.exit.i460:		; preds = %bb.i.i.i.i.i459
	%47 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %5, i32 0, i32 0, i32 1		; <%"struct.Centering<3>"*> [#uses=1]
	%48 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %vint, i32 0, i32 0, i32 1		; <%"struct.Centering<3>"*> [#uses=2]
	%49 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %5, i32 0, i32 0, i32 5		; <%"struct.GuardLayers<3>"*> [#uses=1]
	%50 = bitcast %"struct.GuardLayers<3>"* %49 to i8*		; <i8*> [#uses=1]
	%51 = bitcast %"struct.GuardLayers<3>"* null to i8*		; <i8*> [#uses=2]
	%52 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %3, i32 0, i32 0, i32 1		; <%"struct.Centering<3>"*> [#uses=1]
	%53 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %3, i32 0, i32 0, i32 5		; <%"struct.GuardLayers<3>"*> [#uses=1]
	%54 = bitcast %"struct.GuardLayers<3>"* %53 to i8*		; <i8*> [#uses=1]
	%55 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %2, i32 0, i32 0, i32 1		; <%"struct.Centering<3>"*> [#uses=1]
	%56 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %2, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%57 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %2, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%58 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %2, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%59 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %2, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%60 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %2, i32 0, i32 0, i32 5		; <%"struct.GuardLayers<3>"*> [#uses=1]
	%61 = bitcast %"struct.GuardLayers<3>"* %60 to i8*		; <i8*> [#uses=1]
	%62 = getelementptr %"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >"* %1, i32 0, i32 1, i32 0, i32 0		; <%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=1]
	%63 = getelementptr %"struct.BinaryNode<OpMultiply,Scalar<double>,BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > >"* %1, i32 0, i32 1, i32 1, i32 0		; <%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"*> [#uses=1]
	%64 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, MultiPatchView<GridTag, Remote<Brick>, 3> > > > > >"* null, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%65 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >"* %0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0		; <%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >"*> [#uses=2]
	%66 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >"* %0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%67 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >"* %0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%68 = getelementptr %"struct.Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,ExpressionTag<BinaryNode<OpMultiply, Scalar<double>, BinaryNode<OpAdd, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> >, Field<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >, double, Remote<BrickView> > > > > >"* %0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	br label %bb15

bb15:		; preds = %_Z6assignI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EES5_d13ExpressionTagI10BinaryNodeI10OpMultiply6ScalarIdESD_I5OpAdd9ReferenceI5FieldIS5_dSB_EESL_EEE8OpAssignERKSJ_IT_T0_T1_ESV_RKSJ_IT2_T3_T4_ERKT5_.exit, %_ZN14ScalarCodeInfoILi3ELi6EEC1Ev.exit.i460
	%i.0.reg2mem.0 = phi i32 [ 0, %_ZN14ScalarCodeInfoILi3ELi6EEC1Ev.exit.i460 ], [ %indvar.next, %_Z6assignI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EES5_d13ExpressionTagI10BinaryNodeI10OpMultiply6ScalarIdESD_I5OpAdd9ReferenceI5FieldIS5_dSB_EESL_EEE8OpAssignERKSJ_IT_T0_T1_ESV_RKSJ_IT2_T3_T4_ERKT5_.exit ]		; <i32> [#uses=4]
	call fastcc void @_ZN9CenteringILi3EEC1ERKS0_i(%"struct.Centering<3>"* %47, %"struct.Centering<3>"* %48, i32 %i.0.reg2mem.0) nounwind
	call void @llvm.memcpy.i32(i8* %50, i8* %51, i32 24, i32 4) nounwind
	call fastcc void @_ZN9CenteringILi3EEC1ERKS0_i(%"struct.Centering<3>"* %52, %"struct.Centering<3>"* %48, i32 %i.0.reg2mem.0) nounwind
	call void @llvm.memcpy.i32(i8* %54, i8* %51, i32 24, i32 4) nounwind
	br i1 false, label %bb.i940, label %bb4.i943

bb.i940:		; preds = %bb15
	br label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit944

bb4.i943:		; preds = %bb15
	br label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit944

_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit944:		; preds = %bb4.i943, %bb.i940
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EEEC1Id10MultiPatchIS7_SA_EEERKS_IS5_T_T0_ERK8IntervalILi3EE(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"* null, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* null, %"struct.Interval<3>"* null) nounwind
	call fastcc void @_ZN9CenteringILi3EEC1ERKS0_i(%"struct.Centering<3>"* %55, %"struct.Centering<3>"* null, i32 %i.0.reg2mem.0) nounwind
	call void @llvm.memcpy.i32(i8* %61, i8* %38, i32 24, i32 4) nounwind
	%69 = load %"struct.Loc<3>"** null, align 4		; <%"struct.Loc<3>"*> [#uses=1]
	%70 = ptrtoint %"struct.Loc<3>"* %69 to i32		; <i32> [#uses=1]
	%.off.i911 = sub i32 0, %70		; <i32> [#uses=1]
	%71 = icmp ult i32 %.off.i911, 12		; <i1> [#uses=1]
	%72 = sub i32 0, 0		; <i32> [#uses=2]
	%73 = load i32* %56, align 4		; <i32> [#uses=1]
	%74 = add i32 %73, 0		; <i32> [#uses=1]
	%75 = sub i32 %74, %72		; <i32> [#uses=1]
	%76 = add i32 %75, 0		; <i32> [#uses=1]
	%77 = load i32* null, align 8		; <i32> [#uses=2]
	%78 = load i32* null, align 4		; <i32> [#uses=1]
	%79 = sub i32 %77, %78		; <i32> [#uses=1]
	%80 = load i32* %57, align 4		; <i32> [#uses=1]
	%81 = load i32* null, align 4		; <i32> [#uses=1]
	br i1 %71, label %bb.i912, label %bb4.i915

bb.i912:		; preds = %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit944
	br label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit916

bb4.i915:		; preds = %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit944
	%82 = sub i32 %77, %79		; <i32> [#uses=1]
	%83 = add i32 %82, %80		; <i32> [#uses=1]
	%84 = add i32 %83, %81		; <i32> [#uses=1]
	%85 = load i32* %58, align 8		; <i32> [#uses=2]
	%86 = load i32* null, align 8		; <i32> [#uses=1]
	%87 = sub i32 %85, %86		; <i32> [#uses=2]
	%88 = load i32* %59, align 4		; <i32> [#uses=1]
	%89 = load i32* null, align 4		; <i32> [#uses=1]
	%90 = sub i32 %85, %87		; <i32> [#uses=1]
	%91 = add i32 %90, %88		; <i32> [#uses=1]
	%92 = add i32 %91, %89		; <i32> [#uses=1]
	br label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit916

_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit916:		; preds = %bb4.i915, %bb.i912
	%.0978.0.0.1.0.0.0.0.1.0 = phi i32 [ %84, %bb4.i915 ], [ 0, %bb.i912 ]		; <i32> [#uses=0]
	%.0978.0.0.2.0.0.0.0.0.0 = phi i32 [ %87, %bb4.i915 ], [ 0, %bb.i912 ]		; <i32> [#uses=1]
	%.0978.0.0.2.0.0.0.0.1.0 = phi i32 [ %92, %bb4.i915 ], [ 0, %bb.i912 ]		; <i32> [#uses=0]
	store i32 %72, i32* null, align 8
	store i32 %76, i32* null, align 4
	store i32 %.0978.0.0.2.0.0.0.0.0.0, i32* null, align 8
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EEEC1Id10MultiPatchIS7_SA_EEERKS_IS5_T_T0_ERK8IntervalILi3EE(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"* null, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* null, %"struct.Interval<3>"* null) nounwind
	%93 = load i32* null, align 8		; <i32> [#uses=1]
	%94 = icmp sgt i32 %93, 0		; <i1> [#uses=1]
	br i1 %94, label %bb1.i, label %_Z6assignI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EES5_d13ExpressionTagI10BinaryNodeI10OpMultiply6ScalarIdESD_I5OpAdd9ReferenceI5FieldIS5_dSB_EESL_EEE8OpAssignERKSJ_IT_T0_T1_ESV_RKSJ_IT2_T3_T4_ERKT5_.exit

bb1.i:		; preds = %bb3.i23.i.i, %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit916
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EEED1Ev(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"* %63) nounwind
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EEED1Ev(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"* %62) nounwind
	br label %bb.i17.i14.i

bb.i17.i14.i:		; preds = %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i, %bb1.i
	%i.0.02.rec.i.i.i = phi i32 [ %.rec.i.i.i641, %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i ], [ 0, %bb1.i ]		; <i32> [#uses=1]
	%95 = load double* %64, align 8		; <double> [#uses=1]
	store double %95, double* null, align 8
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEEC1Id14MultiPatchViewI7GridTagS6_I5BrickELi3EEEERKS_IS5_T_T0_ERK5INodeILi3EE(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >"* %65, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatchView<GridTag, Remote<Brick>, 3> >"* null, %"struct.INode<3>"* null) nounwind
	%96 = load %"struct.Loc<3>"** null, align 4		; <%"struct.Loc<3>"*> [#uses=1]
	%97 = ptrtoint %"struct.Loc<3>"* %96 to i32		; <i32> [#uses=1]
	%.off.i21.i.i.i.i = sub i32 0, %97		; <i32> [#uses=1]
	%98 = icmp ult i32 %.off.i21.i.i.i.i, 12		; <i1> [#uses=1]
	br i1 %98, label %bb.i22.i.i.i.i, label %bb3.i25.i.i.i.i

bb.i22.i.i.i.i:		; preds = %bb.i17.i14.i
	%99 = load i32* null, align 4		; <i32> [#uses=1]
	%100 = icmp eq i32 %99, 1		; <i1> [#uses=1]
	%101 = load i32* null, align 4		; <i32> [#uses=1]
	br i1 %100, label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i, label %bb6.i.i24.i.i.i.i

bb6.i.i24.i.i.i.i:		; preds = %bb.i22.i.i.i.i
	br label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i

bb3.i25.i.i.i.i:		; preds = %bb.i17.i14.i
	%102 = load i32* %66, align 8		; <i32> [#uses=2]
	%103 = load i32* %67, align 4		; <i32> [#uses=1]
	%104 = load i32* %68, align 4		; <i32> [#uses=1]
	br label %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i

_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i:		; preds = %bb3.i25.i.i.i.i, %bb6.i.i24.i.i.i.i, %bb.i22.i.i.i.i
	%.rle1279 = phi i32 [ 0, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.rle1277 = phi i32 [ %102, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.rle1275 = phi i32 [ 0, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.01034.0.0.2.0.0.0.0.1.0 = phi i32 [ %104, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.01034.0.0.2.0.0.0.0.0.0 = phi i32 [ 0, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.01034.0.0.1.0.0.0.0.1.0 = phi i32 [ %103, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.01034.0.0.1.0.0.0.0.0.0 = phi i32 [ %102, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.01034.0.0.0.0.0.0.0.1.0 = phi i32 [ 0, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ %101, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%.01034.0.0.0.0.0.0.0.0.0 = phi i32 [ 0, %bb3.i25.i.i.i.i ], [ 0, %bb6.i.i24.i.i.i.i ], [ 0, %bb.i22.i.i.i.i ]		; <i32> [#uses=1]
	%105 = sub i32 %.01034.0.0.0.0.0.0.0.0.0, %.rle1275		; <i32> [#uses=0]
	%106 = sub i32 %.01034.0.0.1.0.0.0.0.0.0, %.rle1277		; <i32> [#uses=0]
	%107 = sub i32 %.01034.0.0.2.0.0.0.0.0.0, %.rle1279		; <i32> [#uses=0]
	store i32 %.01034.0.0.0.0.0.0.0.1.0, i32* null, align 4
	store i32 %.01034.0.0.1.0.0.0.0.1.0, i32* null, align 4
	store i32 %.01034.0.0.2.0.0.0.0.1.0, i32* null, align 4
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEED1Ev(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,Remote<BrickView> >"* %65) nounwind
	%.rec.i.i.i641 = add i32 %i.0.02.rec.i.i.i, 1		; <i32> [#uses=1]
	%108 = load %"struct.INode<3>"** null, align 4		; <%"struct.INode<3>"*> [#uses=1]
	%109 = icmp eq %"struct.INode<3>"* null, %108		; <i1> [#uses=1]
	br i1 %109, label %bb3.i23.i.i, label %bb.i17.i14.i

bb3.i23.i.i:		; preds = %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd6RemoteI9BrickViewEE14physicalDomainEv.exit26.i.i.i.i
	br label %bb1.i

_Z6assignI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EES5_d13ExpressionTagI10BinaryNodeI10OpMultiply6ScalarIdESD_I5OpAdd9ReferenceI5FieldIS5_dSB_EESL_EEE8OpAssignERKSJ_IT_T0_T1_ESV_RKSJ_IT2_T3_T4_ERKT5_.exit:		; preds = %_ZNK11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEE11totalDomainEv.exit916
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 3		; <i1> [#uses=1]
	br i1 %exitcond, label %bb18, label %bb15

bb18:		; preds = %_Z6assignI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd14MultiPatchViewI7GridTag6RemoteI5BrickELi3EES5_d13ExpressionTagI10BinaryNodeI10OpMultiply6ScalarIdESD_I5OpAdd9ReferenceI5FieldIS5_dSB_EESL_EEE8OpAssignERKSJ_IT_T0_T1_ESV_RKSJ_IT2_T3_T4_ERKT5_.exit
	call fastcc void @_ZN11FieldEngineI22UniformRectilinearMeshI10MeshTraitsILi3Ed21UniformRectilinearTag12CartesianTagLi3EEEd10MultiPatchI7GridTag6RemoteI5BrickEEEC1ERKSC_(%"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* null, %"struct.FieldEngine<UniformRectilinearMesh<MeshTraits<3, double, UniformRectilinearTag, CartesianTag, 3> >,double,MultiPatch<GridTag, Remote<Brick> > >"* %42) nounwind
	unreachable
}
