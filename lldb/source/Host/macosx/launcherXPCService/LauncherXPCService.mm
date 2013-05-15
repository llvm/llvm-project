#include <AvailabilityMacros.h>

#if !defined(MAC_OS_X_VERSION_10_7) || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7
#define BUILDING_ON_SNOW_LEOPARD 1
#endif

#if !BUILDING_ON_SNOW_LEOPARD
#define __XPC_PRIVATE_H__
#include <xpc/xpc.h>

// Returns 0 if successful. This is launching as self. No need for further authorization.
int _validate_authorization(xpc_object_t message)
{
    return 0;
}

#endif
