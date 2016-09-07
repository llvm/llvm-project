class Protocol(object):

    def __init__(self, options, config):
        self.options = options
        self.config = config

    def transfer(transfer_specs, dry_run):
        raise "transfer must be overridden by transfer implementation"
