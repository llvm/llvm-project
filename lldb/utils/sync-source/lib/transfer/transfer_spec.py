class TransferSpec(object):

    def __init__(self, source_path, exclude_paths, dest_path):
        self.source_path = source_path
        self.exclude_paths = exclude_paths
        self.dest_path = dest_path

    def __repr__(self):
        fmt = (
            "TransferSpec(source_path='{}', exclude_paths='{}', "
            "dest_path='{}')")
        return fmt.format(self.source_path, self.exclude_paths, self.dest_path)
