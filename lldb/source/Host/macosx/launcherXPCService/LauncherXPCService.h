#ifndef LLDB_LauncherXPCService_h
#define LLDB_LauncherXPCService_h

#define LaunchUsingXPCRightName "com.apple.lldb.LaunchUsingXPC"

// These XPC messaging keys are used for communication between Host.mm and the XPC service.
#define LauncherXPCServiceAuthKey               "auth-key"
#define LauncherXPCServiceArgPrefxKey           "arg"
#define LauncherXPCServiceEnvPrefxKey           "env"
#define LauncherXPCServiceCPUTypeKey            "cpuType"
#define LauncherXPCServicePosixspawnFlagsKey    "posixspawnFlags"
#define LauncherXPCServiceChildPIDKey           "childPID"
#define LauncherXPCServiceErrorTypeKey          "errorType"
#define LauncherXPCServiceCodeTypeKey           "errorCode"

#endif
