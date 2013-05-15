//
//  main.m
//  Used in both LauncherXPCService and LaunchRootXPCService targets
//
//  Copyright (c) 2012 Apple Inc. All rights reserved.
//
#include <AvailabilityMacros.h>

#if !defined(MAC_OS_X_VERSION_10_7) || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7
#define BUILDING_ON_SNOW_LEOPARD 1
#endif

#if !BUILDING_ON_SNOW_LEOPARD
#define __XPC_PRIVATE_H__
#include <xpc/xpc.h>
#include <spawn.h>
#include <signal.h>
#include <assert.h>
#include <sys/errno.h>
#include "LauncherXPCService.h"

// Declaration. Returns 0 if successful.
int _validate_authorization(xpc_object_t message);

// Returns 0 if successful.
int _setup_posixspawn_attributes_file_actions(xpc_object_t message, posix_spawnattr_t *attr, posix_spawn_file_actions_t *file_actions)
{
    *attr = 0;
    
    int errorCode = posix_spawnattr_init(attr);
    if (errorCode)
        return errorCode;
    
    cpu_type_t cpuType = (cpu_type_t)xpc_dictionary_get_int64(message, LauncherXPCServiceCPUTypeKey);
    if (cpuType == -2) {
        cpuType= CPU_TYPE_ANY;
    }
    size_t realCount;
    errorCode = posix_spawnattr_setbinpref_np(attr, 1, &cpuType, &realCount);
    if (errorCode)
        return errorCode;
    
    sigset_t no_signals;
    sigset_t all_signals;
    sigemptyset (&no_signals);
    sigfillset (&all_signals);
    posix_spawnattr_setsigmask(attr, &no_signals);
    posix_spawnattr_setsigdefault(attr, &all_signals);
    
    short flags = xpc_dictionary_get_int64(message, LauncherXPCServicePosixspawnFlagsKey);
    errorCode = posix_spawnattr_setflags(attr, flags);
    if (errorCode)
        return errorCode;
    
    // Setup any file actions. Here we are emulating what debugserver would do normally in Host.mm since the XPC service meant only for debugserver.
    errorCode = posix_spawn_file_actions_init(file_actions);
    if (errorCode)
        return errorCode;
    errorCode = posix_spawn_file_actions_addclose(file_actions, STDIN_FILENO);
    if (errorCode)
        return errorCode;
    errorCode = posix_spawn_file_actions_addclose(file_actions, STDOUT_FILENO);
    if (errorCode)
        return errorCode;
    errorCode = posix_spawn_file_actions_addclose(file_actions, STDERR_FILENO);
    
    return errorCode;
}

bool extract_args(xpc_object_t message, const char *prefix, const char ***argsOut)
{
    char buf[50]; // long enough for 'argXXX'
    memset(buf, 0, 50);
    sprintf(buf, "%sCount", prefix);
    int argsCount = (int)xpc_dictionary_get_int64(message, buf);
    if (argsCount == 0) {
        return true;
    }
    
    const char **argsp = NULL;
    argsp = (const char **)malloc((argsCount+1) * sizeof(argsp[0]));
    if (argsp == NULL) {
        return false;
    }
    
    for (int i=0; i<argsCount; i++) {
        memset(buf, 0, 50);
        sprintf(buf, "%s%i", prefix, i);
        const char *arg = xpc_dictionary_get_string(message, buf);
        argsp[i] = arg;
    }
    argsp[argsCount] = NULL;
    
    *argsOut = argsp;
    return true;
}

// Returns 0 if successful.
int get_args(xpc_object_t message, const char **path, const char ***argsOut, const char ***envOut)
{
    if (!extract_args(message, LauncherXPCServiceArgPrefxKey, argsOut)) {
        return 1;
    }
    *path = (*argsOut)[0];
    
    if (!extract_args(message, LauncherXPCServiceEnvPrefxKey, envOut)) {
        return 2;
    }
    
    return 0;
}

void _wait_for_child_exit(pid_t childPID)
{
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_PROC, childPID, DISPATCH_PROC_EXIT, queue);
    
    if (source) {
        dispatch_source_set_cancel_handler(source, ^{
            dispatch_release(source);
        });
        
        dispatch_source_set_event_handler(source, ^{
            
            // Either finding the process was successful, or the process disappeared before libdispatch got around to hooking up the source.
            dispatch_source_cancel(source);
            
            int status, ret;
            do {
                ret = waitpid(childPID, &status, 0);
            } while (ret < 0 && errno == EINTR);
            
        });
        dispatch_resume(source);
    }
}

static void launcherXPC_peer_event_handler(xpc_connection_t peer, xpc_object_t event) 
{
	xpc_type_t type = xpc_get_type(event);
	if (type == XPC_TYPE_ERROR) {
		if (event == XPC_ERROR_CONNECTION_INVALID) {
			// The client process on the other end of the connection has either
			// crashed or cancelled the connection. After receiving this error,
			// the connection is in an invalid state, and you do not need to
			// call xpc_connection_cancel(). Just tear down any associated state
			// here.
		} else if (event == XPC_ERROR_TERMINATION_IMMINENT) {
			// Handle per-connection termination cleanup.
		}
	} else {
		assert(type == XPC_TYPE_DICTIONARY);
		// Handle the message.
        
        pid_t childPID = 0;
        posix_spawn_file_actions_t file_actions;
        posix_spawnattr_t attributes;
        
        /*
         Types of error. Error code will be specific to each type.
         100 - authorization failure
         101 - posixspawn attributes problem
         102 - get args/env problem
         103 - posixspawn problem
         */
        int errorType = 100;
        int errorCode = _validate_authorization(event);
        if (!errorCode) {
            errorType = 101;
            errorCode = _setup_posixspawn_attributes_file_actions(event, &attributes, &file_actions);
            if (!errorCode) {
                const char *path = NULL;
                const char **argvp = NULL;
                const char **envp = NULL;
                errorType = 102;
                errorCode = get_args(event, &path, &argvp, &envp);
                if (!errorCode) {
                    errorType = 103;
                    errorCode = posix_spawn(&childPID, path, &file_actions, &attributes, (char * const *)argvp, (char * const *)envp);
                    
                    if (argvp) free(argvp);
                    if (envp) free(envp);
                    
                    if (errorCode == 0) {
                        _wait_for_child_exit(childPID);
                    }
                }
            }
        }
        
      	xpc_object_t reply = xpc_dictionary_create_reply(event);
        
        xpc_dictionary_set_int64(reply, LauncherXPCServiceChildPIDKey, childPID);
        if (!childPID) {
            xpc_dictionary_set_int64(reply, LauncherXPCServiceErrorTypeKey, errorType);            
            xpc_dictionary_set_int64(reply, LauncherXPCServiceCodeTypeKey, errorCode);            
        }
        
        xpc_connection_send_message(peer, reply);
        xpc_release(reply);
	}
}

static void launcherXPC_event_handler(xpc_connection_t peer) 
{
	// By defaults, new connections will target the default dispatch
	// concurrent queue.
	xpc_connection_set_event_handler(peer, ^(xpc_object_t event) {
		launcherXPC_peer_event_handler(peer, event);
	});
	
	// This will tell the connection to begin listening for events. If you
	// have some other initialization that must be done asynchronously, then
	// you can defer this call until after that initialization is done.
	xpc_connection_resume(peer);
}

int main(int argc, const char *argv[])
{
	xpc_main(launcherXPC_event_handler);
	return 0;
}
#endif
