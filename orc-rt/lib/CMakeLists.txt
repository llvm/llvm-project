# Support must be configured first: Bedrock embeds its objects.
add_subdirectory(support)
add_subdirectory(bedrock)

# Each connect library builds on the ones listed before it.
add_subdirectory(connect)
add_subdirectory(connect-socket)
add_subdirectory(connect-tcp)
