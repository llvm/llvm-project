import resource


def normalize_limit(limit_value):
    """Normalize RLIM_INFINITY to "infinity" for consistency across Python versions.

    Python 3.15+ returns the platform's max value (e.g., 2^64-1) instead of -1.
    """
    return "infinity" if limit_value == resource.RLIM_INFINITY else str(limit_value)


print("RLIMIT_AS=" + normalize_limit(resource.getrlimit(resource.RLIMIT_AS)[0]))
print("RLIMIT_NOFILE=" + normalize_limit(resource.getrlimit(resource.RLIMIT_NOFILE)[0]))
print("RLIMIT_STACK=" + normalize_limit(resource.getrlimit(resource.RLIMIT_STACK)[0]))
print("RLIMIT_FSIZE=" + normalize_limit(resource.getrlimit(resource.RLIMIT_FSIZE)[0]))
