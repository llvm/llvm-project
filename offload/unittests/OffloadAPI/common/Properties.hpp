#include "Fixtures.hpp"
#include "OffloadAPI.h"

// Properties are grouped by the data type that stores the information. As
// queries for selected properties might have different results for either the
// device or the host, or might be even unsupported for the host, some of the
// tests verify only whether the function that queried the host returned success
// rather than checking the exact value. For tests analyzing the exact value -
// for example, verifying whether the value is greater than the given lower
// bound - only relevant host properties are added to the container with
// properties passed to the fixture during instantiation. The selection of
// relevant properties is achieved by merging or removing selected properties,
// marked as "Relevant" or "Irrelevant" in their names.

inline constexpr size_t MAX_DEVICE_INFO_BYTES = 8;

inline constexpr char zeroArray[MAX_DEVICE_INFO_BYTES] = {};

template <typename T> struct SizedProperty {
  size_t size;
  T property;
};

// Firstly, we list all needed properties explicitly in PropertiesContainer
template <typename T> using PropertiesContainer = std::set<T>;
// Secondly, we process selected properties and save their data sizes for tests
// that require them
template <typename T>
using PropertiesWithSizeContainer = std::vector<SizedProperty<T>>;
template <typename T>
using PropertiesTypes = std::unordered_map<T, SizedProperty<T>>;

template <typename T>
auto createPropertiesWithSizeContainer(
    size_t PropSize, PropertiesContainer<T> SelectedProperties)
    -> PropertiesWithSizeContainer<T> {
  PropertiesWithSizeContainer<T> Res;
  for (auto p : SelectedProperties) {
    Res.push_back({PropSize, p});
  }

  return Res;
}

template <typename T>
auto mergeProperties(
    std::initializer_list<PropertiesWithSizeContainer<T>> properties)
    -> PropertiesWithSizeContainer<T> {
  PropertiesWithSizeContainer<T> finalProperties;

  for (auto prop : properties) {
    finalProperties.insert(finalProperties.end(), prop.begin(), prop.end());
  }

  return finalProperties;
}

template <typename T>
PropertiesContainer<T>
removeIrrelevantProperties(PropertiesContainer<T> base,
                           PropertiesContainer<T> unwanted) {
  PropertiesContainer<T> res(base);

  for (auto prop : unwanted) {
    res.erase(prop);
  }

  return res;
}

template <typename T>
PropertiesTypes<T> inline createTypesMap(
    std::initializer_list<PropertiesWithSizeContainer<T>> properties) {
  PropertiesTypes<T> Res;

  for (auto container : properties) {
    for (auto prop : container) {
      Res.insert({prop.property, prop});
    }
  }

  return Res;
}

// ol_device_info_t
using DeviceInfoProp = PropertiesContainer<ol_device_info_t>;
using DeviceInfoProperties = PropertiesWithSizeContainer<ol_device_info_t>;
using DeviceInfoPropertiesTypes = PropertiesTypes<ol_device_info_t>;

inline const DeviceInfoProp PropBool{OL_DEVICE_INFO_SINGLE_FP_SUPPORT,
                                     OL_DEVICE_INFO_DOUBLE_FP_SUPPORT,
                                     OL_DEVICE_INFO_HALF_FP_SUPPORT};
inline const DeviceInfoProperties BoolProperties =
    createPropertiesWithSizeContainer(sizeof(bool), PropBool);

inline const DeviceInfoProp PropUint32{
    OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
    OL_DEVICE_INFO_MAX_WORK_SIZE,
    OL_DEVICE_INFO_VENDOR_ID,
    OL_DEVICE_INFO_NUM_COMPUTE_UNITS,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
    OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY,
    OL_DEVICE_INFO_MEMORY_CLOCK_RATE,
    OL_DEVICE_INFO_ADDRESS_BITS,
    OL_DEVICE_INFO_NUM_LANES};
inline const DeviceInfoProperties Uint32Properties =
    createPropertiesWithSizeContainer(sizeof(uint32_t), PropUint32);

inline const DeviceInfoProp PropUint64{
    OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, OL_DEVICE_INFO_GLOBAL_MEM_SIZE,
    OL_DEVICE_INFO_WORK_GROUP_LOCAL_MEM_SIZE};
inline const DeviceInfoProperties Uint64Properties =
    createPropertiesWithSizeContainer(sizeof(uint64_t), PropUint64);

inline const DeviceInfoProp PropCapabilitiesFlags{
    OL_DEVICE_INFO_SINGLE_FP_CONFIG, OL_DEVICE_INFO_HALF_FP_CONFIG,
    OL_DEVICE_INFO_DOUBLE_FP_CONFIG};
// sizeof(ol_device_fp_capability_flags_t) == sizeof(uint32_t)
inline const DeviceInfoProperties CapabilitesFlagsProperties =
    createPropertiesWithSizeContainer(sizeof(ol_device_fp_capability_flags_t),
                                      PropCapabilitiesFlags);

inline const DeviceInfoProp PropIrrelevantGTCapabilities = {
    OL_DEVICE_INFO_HALF_FP_CONFIG};
inline const DeviceInfoProp Prop_RelevantGTCapabilites =
    removeIrrelevantProperties(PropCapabilitiesFlags,
                               PropIrrelevantGTCapabilities);
inline const DeviceInfoProperties RelevantGTCapabilitiesProperties =
    createPropertiesWithSizeContainer(sizeof(ol_device_fp_capability_flags_t),
                                      Prop_RelevantGTCapabilites);

inline const DeviceInfoProp PropIrrelevantGTUint32 = {
    OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF};
inline const DeviceInfoProp Prop_RelevantGTUint32 =
    removeIrrelevantProperties(PropUint32, PropIrrelevantGTUint32);
inline const DeviceInfoProperties RelevantGTUint32Properties =
    createPropertiesWithSizeContainer(sizeof(uint32_t), Prop_RelevantGTUint32);

inline const DeviceInfoProp PropDeviceType{OL_DEVICE_INFO_TYPE};
inline const DeviceInfoProperties DeviceTypeProperties =
    createPropertiesWithSizeContainer(sizeof(ol_device_type_t), PropDeviceType);

inline const DeviceInfoProp PropPlatform{OL_DEVICE_INFO_PLATFORM};
inline const DeviceInfoProperties PlatformProperties =
    createPropertiesWithSizeContainer(sizeof(ol_platform_handle_t),
                                      PropPlatform);

inline const DeviceInfoProp PropNames{
    OL_DEVICE_INFO_NAME, OL_DEVICE_INFO_PRODUCT_NAME, OL_DEVICE_INFO_UID,
    OL_DEVICE_INFO_VENDOR, OL_DEVICE_INFO_DRIVER_VERSION};
inline const DeviceInfoProperties NamesProperties =
    createPropertiesWithSizeContainer(0, PropNames);

inline const DeviceInfoProp PropDimensions{
    OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
    OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION};
inline const DeviceInfoProperties DimensionsProperties =
    createPropertiesWithSizeContainer(sizeof(ol_dimensions_t), PropDimensions);

inline const DeviceInfoPropertiesTypes propertiesTypes =
    createTypesMap({BoolProperties, Uint32Properties, Uint64Properties,
                    CapabilitesFlagsProperties});

inline bool defaultCheckIsNonZero(char *buffer) {
  return memcmp(buffer, zeroArray, MAX_DEVICE_INFO_BYTES) != 0;
}

template <typename T>
inline std::string defaultPropertyTestPrinter(
    const ::testing::TestParamInfo<OffloadParam<SizedProperty<T>>> &info) {
  auto device = std::get<0>(info.param);
  auto paramData = std::get<1>(info.param);

  std::string ss;
  llvm::raw_string_ostream finalName(ss);

  auto property = paramData.property;
  finalName << device.Name << "__" << property;

  return SanitizeString(finalName.str());
}

template <typename T>
struct olPropertyTest : OffloadDeviceTestWithParam<SizedProperty<T>> {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(
        OffloadDeviceTestWithParam<SizedProperty<T>>::SetUp());

    auto paramData = this->getTestParam();
    PropertySize = paramData.size;
    Property = paramData.property;
  }

  size_t PropertySize = 0;
  T Property;
};

struct olGetHostDeviceInfoPropertyTest : olPropertyTest<ol_device_info_t> {
  bool isHost() { return Host == this->Device; }
};

struct olGetHostDeviceInfoTest : OffloadDeviceTest {
  void SetUp() override { RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp()); }

  bool isHost() { return Host == this->Device; }
};

// ol_symbol_info
using SymbolInfoTuple = SizedProperty<ol_symbol_info_t>;
using SymbolInfoProp = PropertiesContainer<ol_symbol_info_t>;
using SymbolInfoProperties = PropertiesWithSizeContainer<ol_symbol_info_t>;

inline const SymbolInfoProp PropSymbolInfoGlobal{
    OL_SYMBOL_INFO_KIND, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
    OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE};
inline const SymbolInfoProperties SymbolGlobalProperties{
    {sizeof(ol_symbol_kind_t), OL_SYMBOL_INFO_KIND},
    {sizeof(void *), OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS},
    {sizeof(size_t), OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE}};

struct olGetSymbolInfoSizeGlobalTest
    : OffloadGlobalTestWithParam<SymbolInfoTuple> {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(
        OffloadGlobalTestWithParam<SymbolInfoTuple>::SetUp());

    auto paramData = this->getTestParam();
    PropertySize = paramData.size;
    Property = paramData.property;
  }

  size_t PropertySize = 0;
  ol_symbol_info_t Property;
};

// ol_platform_info_t
inline const ol_platform_info_t PlatformInfoNames[3] = {
    OL_PLATFORM_INFO_NAME, OL_PLATFORM_INFO_VENDOR_NAME,
    OL_PLATFORM_INFO_VERSION};

// ol_alloc_type_t
struct olMemAllocHostOrDeviceTest
    : OffloadDeviceTestWithParam<ol_alloc_type_t> {
  ol_result_t allocateDeviceOrHost(size_t Size, size_t Alignment,
                                   void **Alloc) {
    ol_alloc_type_t AllocType = getTestParam();
    if (AllocType == OL_ALLOC_TYPE_HOST) {
      return olMemAllocAlignedHost(this->Device, Size, Alignment, Alloc);
    }

    return olMemAllocAligned(this->Device, AllocType, Size, Alignment, Alloc);
  }
};

inline const ol_alloc_type_t AllocTypes[3] = {
    OL_ALLOC_TYPE_DEVICE, OL_ALLOC_TYPE_MANAGED, OL_ALLOC_TYPE_HOST};
inline const size_t TestAllocsNum = 1000;

// ol_mem_info_t
using MemInfoProp = PropertiesContainer<ol_mem_info_t>;
using MemInfoProperties = PropertiesWithSizeContainer<ol_mem_info_t>;

inline const MemInfoProp PropMemInfo{OL_MEM_INFO_DEVICE, OL_MEM_INFO_BASE,
                                     OL_MEM_INFO_SIZE, OL_MEM_INFO_TYPE};
inline const MemInfoProperties MemInfoSizeProperties{
    {sizeof(ol_device_handle_t), OL_MEM_INFO_DEVICE},
    {sizeof(void *), OL_MEM_INFO_BASE},
    {sizeof(size_t), OL_MEM_INFO_SIZE},
    {sizeof(ol_alloc_type_t), OL_MEM_INFO_TYPE}};
