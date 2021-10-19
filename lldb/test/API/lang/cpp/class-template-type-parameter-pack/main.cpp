// Named type parameter pack.
template <typename... Is>
struct TypePack { int a; };
TypePack<> emptyTypePack;
TypePack<int> oneElemTypePack;
TypePack<int, float> twoElemTypePack;

// Unnamed type parameter pack.
template <typename... >
struct AnonTypePack { int b; };
AnonTypePack<> emptyAnonTypePack;
AnonTypePack<int> oneElemAnonTypePack;
AnonTypePack<int, float> twoElemAnonTypePack;

// Test type parameter packs combined with non-pack type template parameters.

// Unnamed type parameter pack behind a named type parameter.
template <typename T, typename... >
struct AnonTypePackAfterTypeParam { T c; };
AnonTypePackAfterTypeParam<int> emptyAnonTypePackAfterTypeParam;
AnonTypePackAfterTypeParam<int, float> oneElemAnonTypePackAfterTypeParam;

// Unnamed type parameter pack behind an unnamed type parameter.
template <typename, typename... >
struct AnonTypePackAfterAnonTypeParam { float d; };
AnonTypePackAfterAnonTypeParam<int> emptyAnonTypePackAfterAnonTypeParam;
AnonTypePackAfterAnonTypeParam<int, float> oneElemAnonTypePackAfterAnonTypeParam;

// Named type parameter pack behind an unnamed type parameter.
template <typename, typename... Ts>
struct TypePackAfterAnonTypeParam { int e; };
TypePackAfterAnonTypeParam<int> emptyTypePackAfterAnonTypeParam;
TypePackAfterAnonTypeParam<int, float> oneElemTypePackAfterAnonTypeParam;

// Named type parameter pack behind a named type parameter.
template <typename T, typename... Ts>
struct TypePackAfterTypeParam { int f; };
TypePackAfterTypeParam<int> emptyTypePackAfterTypeParam;
TypePackAfterTypeParam<int, float> oneElemTypePackAfterTypeParam;

// Test type parameter packs combined with non-pack non-type template parameters.

// Unnamed type parameter pack behind a named type parameter.
template <int I, typename... >
struct AnonTypePackAfterNonTypeParam { int g; };
AnonTypePackAfterNonTypeParam<1> emptyAnonTypePackAfterNonTypeParam;
AnonTypePackAfterNonTypeParam<1, int> oneElemAnonTypePackAfterNonTypeParam;

// Unnamed type parameter pack behind an unnamed type parameter.
template <int, typename... >
struct AnonTypePackAfterAnonNonTypeParam { float h; };
AnonTypePackAfterAnonNonTypeParam<1> emptyAnonTypePackAfterAnonNonTypeParam;
AnonTypePackAfterAnonNonTypeParam<1, int> oneElemAnonTypePackAfterAnonNonTypeParam;

// Named type parameter pack behind an unnamed type parameter.
template <int, typename... Ts>
struct TypePackAfterAnonNonTypeParam { int i; };
TypePackAfterAnonNonTypeParam<1> emptyTypePackAfterAnonNonTypeParam;
TypePackAfterAnonNonTypeParam<1, int> oneElemTypePackAfterAnonNonTypeParam;

// Named type parameter pack behind an unnamed type parameter.
template <int I, typename... Ts>
struct TypePackAfterNonTypeParam { int j; };
TypePackAfterNonTypeParam<1> emptyTypePackAfterNonTypeParam;
TypePackAfterNonTypeParam<1, int> oneElemTypePackAfterNonTypeParam;

int main() {
  return 0; // break here
}
