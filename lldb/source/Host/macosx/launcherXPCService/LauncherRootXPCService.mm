#include <AvailabilityMacros.h>

#if !defined(MAC_OS_X_VERSION_10_7) || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7
#define BUILDING_ON_SNOW_LEOPARD 1
#endif

#if !BUILDING_ON_SNOW_LEOPARD
#define __XPC_PRIVATE_H__
#include <xpc/xpc.h>
#include <Security/Security.h>
#include "LauncherXPCService.h"

// Returns 0 if successful.
int _validate_authorization(xpc_object_t message)
{
	size_t data_length = 0ul;
	const char *data_bytes = (const char *)xpc_dictionary_get_data(message, LauncherXPCServiceAuthKey, &data_length);
    
	AuthorizationExternalForm extAuth;
    if (data_length < sizeof(extAuth.bytes))
        return 1;
    
	memcpy(extAuth.bytes, data_bytes, sizeof(extAuth.bytes));
    AuthorizationRef authRef;
	if (AuthorizationCreateFromExternalForm(&extAuth, &authRef) != errAuthorizationSuccess)
        return 2;
    
    AuthorizationItem item1 = { LaunchUsingXPCRightName, 0, NULL, 0 };
    AuthorizationItem items[] = {item1};
    AuthorizationRights requestedRights = {1, items };
    AuthorizationRights *outAuthorizedRights = NULL;
	OSStatus status = AuthorizationCopyRights(authRef, &requestedRights, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &outAuthorizedRights);
	
	// Given a set of rights, return the subset that is currently authorized by the AuthorizationRef given; count(subset) > 0  -> success.
	bool auth_success = (status == errAuthorizationSuccess && outAuthorizedRights && outAuthorizedRights->count > 0) ? true : false;
	if (outAuthorizedRights) AuthorizationFreeItemSet(outAuthorizedRights);
    if (!auth_success)
        return 3;
    
    // On Lion, because the rights initially doesn't exist in /etc/authorization, if an admin user logs in and uses lldb within the first 5 minutes,
    // it is possible to do AuthorizationCopyRights on LaunchUsingXPCRightName and get the rights back.
    // As another security measure, we make sure that the LaunchUsingXPCRightName rights actually exists.
    status = AuthorizationRightGet(LaunchUsingXPCRightName, NULL);
    if (status == errAuthorizationSuccess)
        return 0;
    else
        return 4;
}

#endif
