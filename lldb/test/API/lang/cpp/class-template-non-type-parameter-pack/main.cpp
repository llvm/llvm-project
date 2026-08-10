// Named type parameter pack.
template <int... Is>
struct NonTypePack { int a; };
NonTypePack<> emptyNonTypePack;
NonTypePack<1> oneElemNonTypePack;
NonTypePack<1, 2> twoElemNonTypePack;

// Unnamed type parameter pack.
template <int... >
struct AnonNonTypePack { int b; };
AnonNonTypePack<> emptyAnonNonTypePack;
AnonNonTypePack<1> oneElemAnonNonTypePack;
AnonNonTypePack<1, 2> twoElemAnonNonTypePack;

// Test type parameter packs combined with non-pack type template parameters.

// Unnamed non-type parameter pack behind a named type parameter.
template <typename T, int... >
struct AnonNonTypePackAfterTypeParam { T c; };
AnonNonTypePackAfterTypeParam<int> emptyAnonNonTypePackAfterTypeParam;
AnonNonTypePackAfterTypeParam<int, 1> oneElemAnonNonTypePackAfterTypeParam;

// Unnamed non-type parameter pack behind an unnamed type parameter.
template <typename, int... >
struct AnonNonTypePackAfterAnonTypeParam { float d; };
AnonNonTypePackAfterAnonTypeParam<int> emptyAnonNonTypePackAfterAnonTypeParam;
AnonNonTypePackAfterAnonTypeParam<int, 1> oneElemAnonNonTypePackAfterAnonTypeParam;

// Named non-type parameter pack behind an unnamed type parameter.
template <typename, int... Is>
struct NonTypePackAfterAnonTypeParam { int e; };
NonTypePackAfterAnonTypeParam<int> emptyNonTypePackAfterAnonTypeParam;
NonTypePackAfterAnonTypeParam<int, 1> oneElemNonTypePackAfterAnonTypeParam;

// Named non-type parameter pack behind a named type parameter.
template <typename T, int... Is>
struct NonTypePackAfterTypeParam { int f; };
NonTypePackAfterTypeParam<int> emptyNonTypePackAfterTypeParam;
NonTypePackAfterTypeParam<int, 1> oneElemNonTypePackAfterTypeParam;

// Test non-type parameter packs combined with non-pack non-type template parameters.

// Unnamed non-type parameter pack behind a named non-type parameter.
template <int I, int... >
struct AnonNonTypePackAfterNonTypeParam { int g; };
AnonNonTypePackAfterNonTypeParam<1> emptyAnonNonTypePackAfterNonTypeParam;
AnonNonTypePackAfterNonTypeParam<1, 2> oneElemAnonNonTypePackAfterNonTypeParam;

// Unnamed non-type parameter pack behind an unnamed non-type parameter.
template <int, int... >
struct AnonNonTypePackAfterAnonNonTypeParam { float h; };
AnonNonTypePackAfterAnonNonTypeParam<1> emptyAnonNonTypePackAfterAnonNonTypeParam;
AnonNonTypePackAfterAnonNonTypeParam<1, 2> oneElemAnonNonTypePackAfterAnonNonTypeParam;

// Named non-type parameter pack behind an unnamed non-type parameter.
template <int, int... Is>
struct NonTypePackAfterAnonNonTypeParam { int i; };
NonTypePackAfterAnonNonTypeParam<1> emptyNonTypePackAfterAnonNonTypeParam;
NonTypePackAfterAnonNonTypeParam<1, 2> oneElemNonTypePackAfterAnonNonTypeParam;

// Named non-type parameter pack behind an unnamed non-type parameter.
template <int I, int... Is>
struct NonTypePackAfterNonTypeParam { int j; };
NonTypePackAfterNonTypeParam<1> emptyNonTypePackAfterNonTypeParam;
NonTypePackAfterNonTypeParam<1, 2> oneElemNonTypePackAfterNonTypeParam;

int main() {
  return 0; // break here
}
