import resource

print("RLIMIT_AS=" + str(resource.getrlimit(resource.RLIMIT_AS)[0]))
print("RLIMIT_NOFILE=" + str(resource.getrlimit(resource.RLIMIT_NOFILE)[0]))
print("RLIMIT_STACK=" + str(resource.getrlimit(resource.RLIMIT_STACK)[0]))
print("RLIMIT_FSIZE=" + str(resource.getrlimit(resource.RLIMIT_FSIZE)[0]))
