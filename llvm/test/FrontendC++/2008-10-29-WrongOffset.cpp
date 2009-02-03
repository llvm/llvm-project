// RUN: %llvmgxx %s -c -o /dev/null
// PR2917

#include <complex>
template < int Dim, class T, class EngineTag > class Engine;
template < class Subject, class Sub1, bool SV > struct View1Implementation;
template < class LayoutTag, class PatchTag > struct MultiPatch;
template < class LayoutTag, class PatchTag, int Dim2 > struct MultiPatchView;
template < class Engine, class SubDomain > struct NewEngine
{
};
template < class T > class DomainTraits;
template < class DomT, class T, int Dim > struct DomainTraitsDomain
{
  typedef DomT NewDomain1_t;
};
template < int Dim > class Interval;
template < int Dim > class Loc;
template < class DT > class DomainBase
{
};

template < int Dim, class DT > class Domain:public DomainBase < DT >
{
};
template < int Dim > struct DomainTraits <Interval < Dim >
  >:public DomainTraitsDomain < Interval < Dim >, int, Dim >
{
  enum
  {
    singleValued = false
  };
};
template < class T1 > struct NewDomain1
{
  typedef typename DomainTraits < T1 >::NewDomain1_t SliceType_t;
};
template < class Domain, class Sub > struct TemporaryNewDomain1
{
  typedef typename NewDomain1 < Sub >::SliceType_t SliceType_t;
};
template < int Dim > class Interval:public Domain < Dim,
  DomainTraits < Interval < Dim > > >
{
};
template < int Dim > class GuardLayers
{
};
template < class T > class Observer
{
};

template < class T > class Observable
{
private:T & observed_m;
  int count_m;
};

class RefCounted
{
};
template < class T > class RefCountedPtr
{
public:typedef RefCountedPtr < T > This_t;
  RefCountedPtr (T * const pT):ptr_m (pT)
  {
  }
  inline T *operator-> () const
  {
  }
  T *ptr_m;
};

template < class Dom, class T > class DomainMap
{
};

template < class LayoutTag, int Dim > struct MultiPatchLayoutTraits
{
};
template < int Dim > class LayoutBaseData
{
public:typedef Interval < Dim > Domain_t;
  Domain_t domain_m;
};
template < int Dim, class LBD > class LayoutBase
{
public:typedef LayoutBaseData < Dim > LayoutData_t;
  typedef typename LayoutData_t::Domain_t Domain_t;
  typedef GuardLayers < Dim > GuardLayers_t;
  inline const Domain_t & domain () const
  {
    return pdata_m->domain_m;
  }
  inline const Domain_t & innerDomain () const
  {
  }
  inline GuardLayers_t externalGuards () const
  {
  }
  RefCountedPtr < LBD > pdata_m;
};
template < class Tag > struct Remote;
struct Brick
{
};
template < class Thing, class Sub > struct View1
{
};
template < int Dim, class T, class LayoutTag,
  class PatchTag > struct NewEngine <Engine < Dim, T, MultiPatch < LayoutTag,
  PatchTag > >, Interval < Dim > >
{
  typedef Engine < Dim, T, MultiPatchView < LayoutTag, PatchTag,
    Dim > >Type_t;
};
template < int Dim, class T, class LayoutTag, class PatchTag,
  int Dim2 > struct NewEngine <Engine < Dim, T, MultiPatchView < LayoutTag,
  PatchTag, Dim2 > >, Interval < Dim > >
{
  typedef Engine < Dim, T, MultiPatchView < LayoutTag, PatchTag,
    Dim2 > >Type_t;
};
template < int Dim, class T, class LayoutTag,
  class PatchTag > class Engine < Dim, T, MultiPatch < LayoutTag,
  PatchTag > >:public Observer < typename MultiPatchLayoutTraits < LayoutTag,
  Dim >::Layout_t >
{
public:typedef MultiPatch < LayoutTag, PatchTag > Tag_t;
  typedef Interval < Dim > Domain_t;
};
template < int Dim, class T, class LayoutTag, class PatchTag,
  int Dim2 > class Engine < Dim, T, MultiPatchView < LayoutTag, PatchTag,
  Dim2 > >
{
public:typedef MultiPatchView < LayoutTag, PatchTag, Dim2 > Tag_t;
  typedef Interval < Dim > Domain_t;
  typedef T Element_t;
  enum
  {
    dimensions = Dim
  };
};
class Full;
template < int Dim, class T = double, class EngineTag = Full > class Vector {
};

template < int Dim > inline Interval < Dim >
shrinkRight (const Interval < Dim > &dom, int s)
{
}

template < int Dim > class GridLayout;
struct GridTag
{
};
template < int Dim > struct MultiPatchLayoutTraits <GridTag, Dim >
{
  typedef GridLayout < Dim > Layout_t;
};
template < int Dim > class GridLayoutData:public LayoutBaseData < Dim >,
  public RefCounted, public Observable < GridLayoutData < Dim > >
{
  typedef int AxisIndex_t;
  mutable DomainMap < Interval < 1 >, AxisIndex_t > mapAloc_m[Dim];
};
template < int Dim > class GridLayout:public LayoutBase < Dim,
  GridLayoutData < Dim > >, public Observable < GridLayout < Dim > >,
  public Observer < GridLayoutData < Dim > >
{
public:typedef GridLayout < Dim > This_t;
    GridLayout ();
};
template < class MeshTag, class T, class EngineTag > class Field;
enum CenteringType
{
  VertexType, EdgeType, FaceType, CellType
};
enum ContinuityType
{
  Continuous = 0, Discontinuous
};
template < int Dim > class Centering
{
public:typedef Loc < Dim > Orientation;
  inline int size () const
  {
  }
};
template < int Dim > const Centering < Dim >
canonicalCentering (const enum CenteringType type,
		    const enum ContinuityType discontinuous,
		    const int dimension = 0);
template < class Mesh, class T, class EngineTag > class FieldEngine
{
public:enum
  {
    dimensions = Mesh::dimensions
  };
  enum
  {
    Dim = dimensions
  };
  typedef Engine < Dim, T, EngineTag > Engine_t;
  typedef typename Engine_t::Domain_t Domain_t;
  typedef GuardLayers < Dim > GuardLayers_t;
template < class Layout2 > FieldEngine (const Centering < Dim > &centering, const Layout2 & layout, const Mesh & mesh, int materials = 1):num_materials_m (materials), centering_m (centering),
    stride_m (centering.size ()), physicalCellDomain_m (layout.domain ()),
    guards_m (layout.externalGuards ()), mesh_m (mesh)
  {
  }
  unsigned int num_materials_m;
  Centering < Dim > centering_m;
  int stride_m;
  Domain_t physicalCellDomain_m;
  GuardLayers_t guards_m;
  Mesh mesh_m;
};

template < class Subject > class SubFieldView;
template < class Mesh, class T,
  class EngineTag > class SubFieldView < Field < Mesh, T, EngineTag > >
{
public:typedef Field < Mesh, T, EngineTag > Type_t;
};

template < int Dim, class Mesh, class Domain > struct NewMeshTag
{
  typedef Mesh Type_t;
};
template < class Mesh, class T, class EngineTag,
  class Domain > struct View1Implementation <Field < Mesh, T, EngineTag >,
  Domain, false >
{
  typedef Field < Mesh, T, EngineTag > Subject_t;
  typedef typename Subject_t::Engine_t Engine_t;
  typedef typename NewEngine < Engine_t, Domain >::Type_t NewEngine_t;
  typedef typename NewEngine_t::Element_t NewT_t;
  typedef typename NewEngine_t::Tag_t NewEngineTag_t;
  typedef typename NewMeshTag < NewEngine_t::dimensions, Mesh,
    Domain >::Type_t NewMeshTag_t;
  typedef Field < NewMeshTag_t, NewT_t, NewEngineTag_t > Type_t;
};
template < class Mesh, class T, class EngineTag,
  class Sub1 > struct View1 <Field < Mesh, T, EngineTag >, Sub1 >
{
  typedef Field < Mesh, T, EngineTag > Subject_t;
  typedef typename Subject_t::Domain_t Domain_t;
  typedef TemporaryNewDomain1 < Domain_t, Sub1 > NewDomain_t;
  typedef typename NewDomain_t::SliceType_t SDomain_t;
  enum
  {
    sv = DomainTraits < SDomain_t >::singleValued
  };
  typedef View1Implementation < Subject_t, SDomain_t, sv > Dispatch_t;
  typedef typename Dispatch_t::Type_t Type_t;
};
template < class Mesh, class T = double, class EngineTag = Brick > class Field {
public:typedef Mesh MeshTag_t;
  typedef Mesh Mesh_t;
  typedef Field < Mesh, T, EngineTag > This_t;
  typedef FieldEngine < Mesh, T, EngineTag > FieldEngine_t;
  enum
  {
    dimensions = FieldEngine_t::dimensions
  };
  typedef Engine < dimensions, T, EngineTag > Engine_t;
  typedef typename Engine_t::Domain_t Domain_t;
  typedef Centering < dimensions > Centering_t;
  template < class Layout2 > Field (const Centering_t & centering,
				    const Layout2 & layout,
				    const Mesh_t &
				    mesh):fieldEngine_m (centering, layout,
							 mesh)
  {
  }
  inline typename SubFieldView < This_t >::Type_t center (int c) const
  {
  }
  inline typename View1 < This_t, Domain_t >::Type_t all () const
  {
  }
  template < class T1 > const This_t & operator= (const T1 & rhs) const
  {
  }
private:  FieldEngine_t fieldEngine_m;
};

struct UniformRectilinearTag
{
};
struct CartesianTag
{
};
template < class MeshTraits > struct CartesianURM;
template < class MeshTraits > class UniformRectilinearMeshData;
template < class MeshTraits > class UniformRectilinearMesh;
template < int Dim, typename T = double, class MeshTag =
  UniformRectilinearTag, class CoordinateSystemTag = CartesianTag, int CDim =
  Dim > struct MeshTraits;
template < int Dim, typename T, class MeshTag, class CoordinateSystemTag,
  int CDim > struct MeshTraitsBase
{
  typedef MeshTraits < Dim, T, MeshTag, CoordinateSystemTag,
    CDim > MeshTraits_t;
  enum
  {
    dimensions = Dim
  };
  typedef Vector < CDim, T > PointType_t;
};
template < int Dim, typename T, int CDim > struct MeshTraits <Dim, T,
  UniformRectilinearTag, CartesianTag, CDim >:public MeshTraitsBase < Dim, T,
  UniformRectilinearTag, CartesianTag, CDim >
{
  typedef typename MeshTraitsBase < Dim, T, UniformRectilinearTag,
    CartesianTag, CDim >::MeshTraits_t MeshTraits_t;
  typedef CartesianURM < MeshTraits_t > CoordinateSystem_t;
  typedef UniformRectilinearMeshData < MeshTraits_t > MeshData_t;
  typedef UniformRectilinearMesh < MeshTraits_t > Mesh_t;
  typedef Vector < CDim, T > SpacingsType_t;
};
template < int Dim > class NoMeshData:public RefCounted
{
public:NoMeshData ()
  {
  }
  template < class Layout >
    explicit NoMeshData (const Layout &
			 layout):physicalVertexDomain_m (layout.
							 innerDomain ()),
    physicalCellDomain_m (shrinkRight (physicalVertexDomain_m, 1)),
    totalVertexDomain_m (layout.domain ()),
    totalCellDomain_m (shrinkRight (totalVertexDomain_m, 1))
  {
  }
private:Interval < Dim > physicalVertexDomain_m, physicalCellDomain_m;
  Interval < Dim > totalVertexDomain_m, totalCellDomain_m;
};

template < class MeshTraits > class UniformRectilinearMeshData:public NoMeshData <
  MeshTraits::
  dimensions >
{
public:typedef typename
    MeshTraits::MeshData_t
    MeshData_t;
  typedef typename
    MeshTraits::PointType_t
    PointType_t;
  typedef typename
    MeshTraits::SpacingsType_t
    SpacingsType_t;
  enum
  {
    dimensions = MeshTraits::dimensions
  };
  template < class Layout > UniformRectilinearMeshData (const Layout & layout,
							const PointType_t &
							origin,
							const SpacingsType_t &
							spacings):
    NoMeshData <
  dimensions > (layout),
  origin_m (origin),
  spacings_m (spacings)
  {
  }
private:PointType_t origin_m;
  SpacingsType_t
    spacings_m;
};

template < class MeshTraits > class UniformRectilinearMesh:public MeshTraits::
  CoordinateSystem_t
{
public:typedef MeshTraits
    MeshTraits_t;
  typedef typename
    MeshTraits::MeshData_t
    MeshData_t;
  typedef typename
    MeshTraits::PointType_t
    PointType_t;
  typedef typename
    MeshTraits::SpacingsType_t
    SpacingsType_t;
  enum
  {
    dimensions = MeshTraits::dimensions
  };
  template < class Layout >
    inline UniformRectilinearMesh (const Layout & layout,
				   const PointType_t & origin,
				   const SpacingsType_t & spacings):
  data_m (new MeshData_t (layout, origin, spacings))
  {
  }
private:RefCountedPtr < MeshData_t > data_m;
};

template < class MeshTraits > struct GenericURM
{
};
template < class MeshTraits > struct CartesianURM:
  public
  GenericURM <
  MeshTraits >
{
};
template < int
  dim,
  class
  MeshTag = UniformRectilinearTag, class CoordinateSystemTag = CartesianTag > struct ParallelTraits {
  enum
  {
    Dim = dim
  };
  typedef
    GridLayout <
    dim >
    Layout_t;
  typedef
    MeshTraits <
    dim, double,
    MeshTag,
    CoordinateSystemTag >
    MeshTraits_t;
  typedef typename
    MeshTraits_t::Mesh_t
    Mesh_t;
  typedef
    MultiPatch <
    GridTag,
    Remote <
  Brick > >
    Engine_t;
};
template < class ComputeTraits > struct RhalkTraits:
  public
  ComputeTraits
{
  typedef typename
    ComputeTraits::Mesh_t
    Mesh_t;
  typedef typename
    ComputeTraits::Engine_t
    Engine_t;
  enum
  {
    Dim = ComputeTraits::Dim
  };
  typedef
    Centering <
    Dim >
    Centering_t;
  typedef typename
    Mesh_t::SpacingsType_t
    Spacings_t;
  typedef
    Field <
    Mesh_t, double,
    Engine_t >
    Scalar_t;
};
enum
{
  Dim = 3
};
typedef
  RhalkTraits <
  ParallelTraits <
  Dim,
  UniformRectilinearTag,
CartesianTag > >
  Traits_t;
Vector < Dim > origin;
Traits_t::Spacings_t spacings;
int
main (int argc, char **argv)
{
  Traits_t::Layout_t layout;
  Traits_t::Mesh_t mesh (layout, origin, spacings);
  Traits_t::Centering_t face =
    canonicalCentering < Traits_t::Dim > (FaceType, Continuous);
  Traits_t::Scalar_t v (face, layout, mesh);
  for (int i = 0; i < Dim; ++i)
    v.center (i).all () = std::numeric_limits < double >::signaling_NaN ();
}
