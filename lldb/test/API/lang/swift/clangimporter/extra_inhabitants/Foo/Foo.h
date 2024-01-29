typedef struct {} *StructPtr;
// Simulates the CoreFoundation CF_BRIDGED_TYPE(id) macro.
typedef struct  __attribute__((objc_bridge(id))) {} *BridgedPtr;
typedef id OpaqueObj;
typedef void *VoidPtr;
