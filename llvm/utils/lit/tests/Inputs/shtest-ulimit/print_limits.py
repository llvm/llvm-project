import resource

print("RLIMIT_AS=" + str(resource.getrlimit(resource.RLIMIT_AS)[0]))
print("RLIMIT_NOFILE=" + str(resource.getrlimit(resource.RLIMIT_NOFILE)[0]))
